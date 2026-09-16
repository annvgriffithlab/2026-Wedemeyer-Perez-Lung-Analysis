"""
Parenchymal Inflammation Quantification — Lung H&E Histology  (v3)
===================================================================
Analyzes H&E-stained lung tissue images to quantify inflammatory cell infiltration.

v3 Changes
-----------
  * Group detection: filenames starting with "WT" or "FGF" are automatically grouped.
    Customize with --groups (e.g. --groups WT FGF).
  * Group statistics figure: mean +/- SD bars with individual data points overlaid,
    plus Welch's t-test p-value -- publication-ready.
  * group_stats.csv: one row per group with mean, SD, SEM, n, and p-value.
  * Handles filenames with spaces (always quote paths on Windows, see Usage).

v2 Change: Fixed nuclear threshold derived from a reference (WT) image
---------------------------------------------------------------------------
The original script used per-image Otsu thresholding on the B-R channel.
This is unreliable when experimental conditions (e.g. FGF treatment) reduce
hematoxylin staining intensity -- the B-R histogram loses its bimodality and
Otsu finds a spuriously low threshold, inflating nuclear detection.

Solution: calibrate the nuclear threshold from a reference image with confirmed
strong hematoxylin staining (typically a WT/control sample), then apply that
fixed value uniformly to all images. A bimodality QC score is computed for every
image and flagged in the CSV when the distribution is poorly separated.

Method:
  1. Background/airspace segmentation via Otsu thresholding on brightness (per-image)
  2. Nuclear segmentation via FIXED B-R threshold derived from reference image
  3. Morphological cleanup (opening + small object removal)
  4. Connected component classification by area and shape (eccentricity)
     - Inflammatory cells: small (16-800 px at 10x), relatively round (eccentricity < 0.95)
     - Structural/wall nuclei: large or elongated

Output:
  - results.csv           : per-image metrics (includes group and bimodality QC)
  - group_stats.csv       : group means, SD, SEM, n, t-test p-value
  - group_comparison.png  : mean +/- SD bar chart with individual points + p-value
  - validation/           : per-image validation figures for PI review

Usage (quote any path that contains spaces):
  Windows:
    python lung_inflammation_analysis.py ^
        --input "C:\\Images" ^
        --output "C:\\Results" ^
        --reference "C:\\Images\\WT Aged1 Stitched 10x.tif"

  Mac/Linux:
    python lung_inflammation_analysis.py \\
        --input "/data/images" \\
        --output "/data/results" \\
        --reference "/data/images/WT Aged1 Stitched 10x.tif"

  Reuse a saved threshold (skip re-calibration):
    python lung_inflammation_analysis.py --input ... --output ... --nuclear-threshold 17.12

Requirements:
  pip install numpy scipy scikit-image pillow matplotlib opencv-python pandas tqdm
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage import filters, measure, morphology
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        write = staticmethod(print)
        def __init__(self, iterable, desc=""):
            self._it = iterable
            self._desc = desc
        def __iter__(self):
            items = list(self._it)
            for i, item in enumerate(items):
                yield item


# ── Tunable parameters ────────────────────────────────────────────────────────

PARAMS = {
    # ── Nuclear detection ─────────────────────────────────────────────────
    # Minimum nuclear object size (pixels). Objects smaller than this are noise.
    # At 10x magnification, a lymphocyte nucleus ≈ 16–50 px.
    "min_nuclear_size": 16,

    # Maximum single nuclear object size (pixels) — caps runaway detections.
    "max_nuclear_size": 800,

    # Morphological disk radius for binary opening (noise removal before clustering).
    "morph_disk_radius": 1,

    # ── Cluster detection ─────────────────────────────────────────────────
    # Dilation radius (pixels) applied to the nuclear mask before merging into
    # clusters. Nuclei within 2 × dilation_radius of each other will merge.
    # At 10x: 8 px ≈ 16 µm gap — captures tightly packed inflammatory infiltrates
    # while keeping well-separated structural nuclei as isolated objects.
    "cluster_dilation_radius": 8,

    # Minimum number of individual nuclei required within a cluster to be counted
    # as a true inflammatory infiltrate. Filters out isolated single cells and
    # small doublets that don't represent genuine parenchymal inflammation.
    "min_nuclei_per_cluster": 15,

    # Minimum total pixel area of a qualifying cluster (secondary size filter).
    "min_cluster_area_px": 200,
}



# ── Calibration & QC helpers ──────────────────────────────────────────────────

def calibrate_nuclear_threshold(reference_path: Path) -> float:
    """
    Derive the nuclear B−R threshold from a reference image using Otsu.

    The reference should be a WT/control slide with strong, confirmed hematoxylin
    staining so the B−R histogram is clearly bimodal. The resulting threshold is
    then applied as a FIXED value to all other images in the batch.

    Returns: float — the Otsu B−R threshold from the reference image.
    """
    img = Image.open(reference_path)
    arr = np.array(img)
    R = arr[:, :, 0].astype(float)
    G = arr[:, :, 1].astype(float)
    B = arr[:, :, 2].astype(float)
    brightness = (R + G + B) / 3.0

    otsu_bg = float(filters.threshold_otsu(brightness))
    tissue_mask = brightness <= otsu_bg

    B_minus_R = B - R
    br_tissue_vals = B_minus_R[tissue_mask].astype(float)
    threshold = float(filters.threshold_otsu(br_tissue_vals))

    bim = bimodality_score(br_tissue_vals, threshold)
    print(f"\n  Reference image : {reference_path.name}")
    print(f"  Nuclear threshold (B−R > {threshold:.2f}) derived from reference")
    print(f"  Bimodality score : {bim:.3f}  {'✓ good separation' if bim >= 0.2 else '⚠ weak separation — check reference image'}")
    return threshold


def bimodality_score(br_tissue_vals: np.ndarray, threshold: float) -> float:
    """
    Quantify how well the B−R histogram separates into two peaks at `threshold`.

    Score = (mean_above - mean_below) / (std_above + std_below + 1e-6)

    Interpretation:
      >= 0.5  : excellent separation — threshold is reliable
      0.2–0.5 : acceptable
      < 0.2   : poor bimodality — treat results with caution (flagged in CSV)
    """
    below = br_tissue_vals[br_tissue_vals <= threshold]
    above = br_tissue_vals[br_tissue_vals >  threshold]
    if len(below) < 10 or len(above) < 10:
        return 0.0
    score = (above.mean() - below.mean()) / (above.std() + below.std() + 1e-6)
    return float(score)


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_image(path: Path, nuclear_threshold: float, params: dict = PARAMS) -> dict:
    """
    Run full inflammation analysis on a single H&E TIF image.

    Args:
        path              : path to the .tif image
        nuclear_threshold : fixed B−R threshold derived from the reference (WT) image.
                            Applied uniformly to all images so detection criteria are
                            identical regardless of per-sample staining variation.
        params            : size / shape classification parameters

    Returns a dict of metrics plus intermediate arrays for validation figures.
    """
    img = Image.open(path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    total_pixels = h * w

    R = arr[:, :, 0].astype(float)
    G = arr[:, :, 1].astype(float)
    B = arr[:, :, 2].astype(float)
    brightness = (R + G + B) / 3.0

    # ── Step 1: Background / airspace segmentation ─────────────────────────
    # Otsu threshold on brightness — still per-image (illumination can vary)
    otsu_bg = float(filters.threshold_otsu(brightness))
    bg_mask    = brightness > otsu_bg
    tissue_mask = ~bg_mask
    tissue_pixels = int(tissue_mask.sum())

    # ── Step 2: Nuclear segmentation ──────────────────────────────────────
    # Blue-minus-Red channel highlights hematoxylin-stained nuclei.
    # FIXED threshold from reference image — not recomputed per sample.
    B_minus_R = (B - R)

    # QC: compute bimodality score for this image at the fixed threshold
    br_tissue_vals = B_minus_R[tissue_mask].astype(float)
    bim_score = bimodality_score(br_tissue_vals, nuclear_threshold)
    bim_flag  = bim_score < 0.2   # True = warn in CSV

    nuclear_mask = (
        tissue_mask &
        (B_minus_R > nuclear_threshold) &
        (brightness < 210)
    )

    # ── Step 3: Morphological cleanup of individual nuclei ────────────────
    disk_clean = morphology.disk(params["morph_disk_radius"])
    nuclear_clean = morphology.opening(nuclear_mask, disk_clean)
    nuclear_clean = morphology.remove_small_objects(
        nuclear_clean, max_size=params["min_nuclear_size"] - 1
    )
    # Also remove implausibly large single-nucleus blobs
    nuclear_clean = morphology.remove_small_objects(
        ~morphology.remove_small_objects(~nuclear_clean, max_size=params["max_nuclear_size"])
    ) if params["max_nuclear_size"] else nuclear_clean

    # Label individual nuclei (used to count nuclei-per-cluster below)
    labeled_nuclei = measure.label(nuclear_clean)
    nuclear_props  = measure.regionprops(labeled_nuclei)
    total_nuclei   = len(nuclear_props)

    # ── Step 4: Cluster detection ──────────────────────────────────────────
    # Dilate the nuclear mask so nearby nuclei merge into spatial clusters.
    # Each connected component of the dilated mask is a candidate infiltrate.
    dilation_disk   = morphology.disk(params["cluster_dilation_radius"])
    dilated_nuclear = morphology.dilation(nuclear_clean, dilation_disk)

    labeled_clusters = measure.label(dilated_nuclear)
    cluster_props    = measure.regionprops(labeled_clusters)

    # For each candidate cluster, count how many original nuclei fall inside it.
    # A nucleus belongs to a cluster if its centroid lies within the dilated blob.
    cluster_nucleus_count = {}
    for np_ in nuclear_props:
        cy, cx = int(np_.centroid[0]), int(np_.centroid[1])
        cid = labeled_clusters[cy, cx]
        if cid > 0:
            cluster_nucleus_count[cid] = cluster_nucleus_count.get(cid, 0) + 1

    # Keep clusters meeting both the nucleus-count and area criteria
    infl_cluster_ids  = set()
    reject_cluster_ids = set()
    for cp in cluster_props:
        n_nuclei   = cluster_nucleus_count.get(cp.label, 0)
        qualifies  = (
            n_nuclei  >= params["min_nuclei_per_cluster"] and
            cp.area   >= params["min_cluster_area_px"]
        )
        if qualifies:
            infl_cluster_ids.add(cp.label)
        else:
            reject_cluster_ids.add(cp.label)

    # Build pixel masks for qualifying clusters (use UNDILATED nuclear pixels
    # for area measurement so we report actual hematoxylin area, not inflated)
    infl_mask   = np.zeros(arr.shape[:2], bool)
    reject_mask = np.zeros(arr.shape[:2], bool)
    for np_ in nuclear_props:
        cy, cx = int(np_.centroid[0]), int(np_.centroid[1])
        cid = labeled_clusters[cy, cx]
        if cid in infl_cluster_ids:
            infl_mask[labeled_nuclei == np_.label] = True
        elif cid in reject_cluster_ids or cid == 0:
            reject_mask[labeled_nuclei == np_.label] = True

    infl_pixels    = int(infl_mask.sum())
    n_infl_clusters  = len(infl_cluster_ids)
    n_reject_clusters = len(reject_cluster_ids)

    # Average nuclei per qualifying cluster (useful QC metric)
    avg_nuclei_per_cluster = (
        np.mean([cluster_nucleus_count.get(cid, 0) for cid in infl_cluster_ids])
        if infl_cluster_ids else 0.0
    )

    return {
        # ── Scalar metrics (written to CSV) ──────────────────────────────
        "file":                      path.name,
        "image_width_px":            w,
        "image_height_px":           h,
        "total_pixels":              total_pixels,
        "tissue_pixels":             tissue_pixels,
        "airspace_pixels":           int(bg_mask.sum()),
        "tissue_pct":                round(100 * tissue_pixels / total_pixels, 2),
        "airspace_pct":              round(100 * bg_mask.sum() / total_pixels, 2),
        "otsu_brightness_thresh":    round(otsu_bg, 2),
        "nuclear_thresh_fixed":      round(nuclear_threshold, 2),
        "bimodality_score":          round(bim_score, 3),
        "bimodality_flag":           "WARN — weak H staining, check overlay" if bim_flag else "OK",
        "total_nuclei_detected":     total_nuclei,
        "candidate_clusters":        len(cluster_props),
        "infl_clusters":             n_infl_clusters,
        "rejected_clusters":         n_reject_clusters,
        "avg_nuclei_per_infl_cluster": round(float(avg_nuclei_per_cluster), 1),
        "infl_pixels":               infl_pixels,
        "infl_pct_of_tissue":        round(100 * infl_pixels / tissue_pixels, 3) if tissue_pixels else 0,
        "infl_pct_of_total":         round(100 * infl_pixels / total_pixels, 3),

        # ── Intermediate arrays (for validation figures only) ─────────────
        "_arr":               arr,
        "_bg_mask":           bg_mask,
        "_tissue_mask":       tissue_mask,
        "_nuclear_clean":     nuclear_clean,
        "_dilated_nuclear":   dilated_nuclear,
        "_infl_mask":         infl_mask,
        "_reject_mask":       reject_mask,
        "_labeled_clusters":  labeled_clusters,
        "_infl_cluster_ids":  infl_cluster_ids,
        "_B_minus_R":         B_minus_R,
        "_brightness":        brightness,
        "_otsu_bg":           otsu_bg,
        "_otsu_nuclear":      nuclear_threshold,
        "_br_tissue_vals":    br_tissue_vals,
        "_bim_score":         bim_score,
        "_bim_flag":          bim_flag,
    }



# ── Validation figure ─────────────────────────────────────────────────────────

def make_validation_figure(result: dict, out_path: Path, downsample: float = 0.12):
    """
    Generate a 6-panel validation figure for PI review.

    Panels:
      1. Original image
      2. Brightness channel + Otsu background threshold
      3. Individual nuclei detected (pre-clustering)
      4. Dilated clusters — all candidates
      5. Qualifying inflammatory clusters only (overlaid on original)
      6. Metrics summary text
    """
    arr             = result["_arr"]
    bg_mask         = result["_bg_mask"]
    nuclear_clean   = result["_nuclear_clean"]
    dilated_nuclear = result["_dilated_nuclear"]
    infl_mask       = result["_infl_mask"]
    reject_mask     = result["_reject_mask"]
    brightness      = result["_brightness"]
    B_minus_R       = result["_B_minus_R"]
    otsu_bg         = result["_otsu_bg"]
    otsu_nuclear    = result["_otsu_nuclear"]

    h_full, w_full = arr.shape[:2]
    w_s = int(w_full * downsample)
    h_s = int(h_full * downsample)

    def rs(im_arr):
        return np.array(Image.fromarray(im_arr).resize((w_s, h_s)))

    def rs_mask(mask):
        return np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize((w_s, h_s))
        ) > 127

    thumb         = rs(arr)
    bg_s          = rs_mask(bg_mask)
    nuclear_s     = rs_mask(nuclear_clean)
    dilated_s     = rs_mask(dilated_nuclear)
    infl_s        = rs_mask(infl_mask)
    reject_s      = rs_mask(reject_mask)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0d1117")
    TITLE_KW = dict(color="white", fontsize=11, fontweight="bold", pad=6)
    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    # Panel 1 — Original
    axes[0, 0].imshow(thumb)
    axes[0, 0].set_title(f"1. Original — {result['file']}", **TITLE_KW)

    # Panel 2 — Brightness + background Otsu
    bright_s = np.array(Image.fromarray(brightness.astype(np.uint8)).resize((w_s, h_s)))
    bg_rgba  = np.zeros((*bright_s.shape, 4))
    bg_rgba[bg_s] = [0.8, 0.3, 0.3, 0.35]
    axes[0, 1].imshow(bright_s, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].imshow(bg_rgba)
    axes[0, 1].set_title(
        f"2. Brightness — airspace mask\nOtsu threshold = {otsu_bg:.1f}", **TITLE_KW
    )

    # Panel 3 — Individual nuclei detected (pre-clustering)
    nuc_overlay = thumb.copy()
    nuc_overlay[nuclear_s] = [255, 230, 50]   # yellow = detected nuclei
    axes[0, 2].imshow(nuc_overlay)
    axes[0, 2].set_title(
        f"3. Individual nuclei (pre-cluster)\n"
        f"Fixed B\u2212R threshold = {otsu_nuclear:.1f}  |  "
        f"Bimodality = {result['_bim_score']:.3f}{'  \u26a0' if result['_bim_flag'] else '  \u2713'}  |  "
        f"n = {result['total_nuclei_detected']:,}",
        **TITLE_KW
    )

    # Panel 4 — All candidate clusters (dilated, pre-filter)
    clust_overlay = thumb.copy()
    clust_overlay[dilated_s & ~infl_s] = [200, 100, 200]  # purple = rejected clusters
    clust_overlay[dilated_s &  infl_s] = [40,  220,  80]  # green  = qualifying clusters
    axes[1, 0].imshow(clust_overlay)
    axes[1, 0].set_title(
        f"4. Cluster detection\n"
        f"Green = qualifying (\u2265{PARAMS['min_nuclei_per_cluster']} nuclei)   "
        f"Purple = rejected",
        **TITLE_KW
    )
    patches = [
        mpatches.Patch(color=[40/255, 220/255, 80/255],   label=f"Infl. cluster ({result['infl_clusters']})"),
        mpatches.Patch(color=[200/255, 100/255, 200/255], label=f"Rejected cluster ({result['rejected_clusters']})"),
    ]
    axes[1, 0].legend(handles=patches, loc="lower right", fontsize=8,
                      facecolor="#111", labelcolor="white", framealpha=0.85)

    # Panel 5 — Qualifying inflammation overlaid on original
    axes[1, 1].imshow(thumb, alpha=0.5)
    infl_rgba = np.zeros((*infl_s.shape, 4))
    infl_rgba[infl_s] = [0.15, 0.85, 0.30, 0.90]
    axes[1, 1].imshow(infl_rgba)
    axes[1, 1].set_title(
        f"5. Inflammatory clusters (final mask)\n"
        f"{result['infl_clusters']} clusters   "
        f"avg {result['avg_nuclei_per_infl_cluster']:.0f} nuclei/cluster",
        **TITLE_KW
    )

    # Panel 6 — Metrics summary
    ax6 = axes[1, 2]
    ax6.set_facecolor("#111827")
    metrics_text = (
        f"QUANTIFICATION SUMMARY\n"
        f"{'─'*36}\n"
        f"File:               {result['file']}\n"
        f"Image size:         {result['image_width_px']} \u00d7 {result['image_height_px']} px\n\n"
        f"TISSUE COMPOSITION\n"
        f"  Tissue area:      {result['tissue_pct']:.1f}%\n"
        f"  Airspace:         {result['airspace_pct']:.1f}%\n\n"
        f"THRESHOLDS\n"
        f"  Background:       brightness > {result['otsu_brightness_thresh']:.1f}  (Otsu, per-image)\n"
        f"  Nuclear (B\u2212R):    B\u2212R > {result['nuclear_thresh_fixed']:.1f}  (FIXED from reference)\n\n"
        f"STAINING QC\n"
        f"  Bimodality score: {result['bimodality_score']:.3f}\n"
        f"  Status:           {result['bimodality_flag']}\n\n"
        f"NUCLEI & CLUSTERS\n"
        f"  Total nuclei:     {result['total_nuclei_detected']:,}\n"
        f"  Candidate clusters: {result['candidate_clusters']}\n"
        f"  Qualifying (infl): {result['infl_clusters']}  (>= {PARAMS['min_nuclei_per_cluster']} nuclei,\n"
        f"                              >= {PARAMS['min_cluster_area_px']} px)\n"
        f"  Rejected:         {result['rejected_clusters']}\n"
        f"  Avg nuclei/cluster: {result['avg_nuclei_per_infl_cluster']:.1f}\n\n"
        f"INFLAMMATION\n"
        f"  % of tissue:      {result['infl_pct_of_tissue']:.2f}%\n"
        f"  % of total image: {result['infl_pct_of_total']:.2f}%\n"
        f"  Pixel count:      {result['infl_pixels']:,}\n"
    )
    ax6.text(
        0.05, 0.97, metrics_text,
        transform=ax6.transAxes,
        va="top", ha="left",
        fontsize=8.0,
        fontfamily="monospace",
        color="white",
        linespacing=1.5,
    )
    ax6.set_title("6. Metrics summary", **TITLE_KW)

    plt.suptitle(
        "Validation Report — Cluster-Based Parenchymal Inflammation Quantification",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()



# ── Threshold sensitivity report ─────────────────────────────────────────────

def make_threshold_sensitivity_figure(result: dict, out_path: Path):
    """
    Show how % inflammation changes as cluster_dilation_radius and
    min_nuclei_per_cluster vary. Key validation plot for PI.
    """
    nuclear_clean   = result["_nuclear_clean"]
    tissue_px       = result["tissue_pixels"]

    dilation_range  = range(2, 18, 2)          # dilation radii to test
    min_nuclei_range = [2, 3, 5, 7, 10, 15]   # min-nuclei-per-cluster values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # Panel 1: vary dilation radius (fix min_nuclei at default)
    default_min_n = PARAMS["min_nuclei_per_cluster"]
    default_min_area = PARAMS["min_cluster_area_px"]
    pcts_dilation = []
    for dr in dilation_range:
        dil = morphology.dilation(nuclear_clean, morphology.disk(dr))
        lbl = measure.label(dil)
        # count nuclei per cluster
        lbl_nuc = measure.label(nuclear_clean)
        nuc_props = measure.regionprops(lbl_nuc)
        cnts: dict = {}
        for np_ in nuc_props:
            cy, cx = int(np_.centroid[0]), int(np_.centroid[1])
            cid = lbl[cy, cx]
            if cid > 0:
                cnts[cid] = cnts.get(cid, 0) + 1
        infl_px = sum(
            np_.area for np_ in nuc_props
            if cnts.get(lbl[int(np_.centroid[0]), int(np_.centroid[1])], 0) >= default_min_n
            and measure.regionprops(lbl)[lbl[int(np_.centroid[0]), int(np_.centroid[1])]-1].area >= default_min_area
            if lbl[int(np_.centroid[0]), int(np_.centroid[1])] > 0
        )
        pcts_dilation.append(100 * infl_px / tissue_px if tissue_px else 0)

    axes[0].plot(list(dilation_range), pcts_dilation, color="#3a86ff", linewidth=2.5, marker="o", markersize=5)
    axes[0].axvline(PARAMS["cluster_dilation_radius"], color="#ff6b6b", linestyle="--",
                    linewidth=1.5, label=f"Default ({PARAMS['cluster_dilation_radius']} px)")
    axes[0].set_xlabel("Cluster dilation radius (px)", fontsize=11)
    axes[0].set_ylabel("% Parenchymal inflammation (of tissue)", fontsize=11)
    axes[0].set_title(
        f"Sensitivity to dilation radius\n(min nuclei/cluster fixed at {default_min_n})",
        color="white", fontsize=11
    )
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=9)

    # Panel 2: vary min_nuclei_per_cluster (fix dilation at default)
    default_dr = PARAMS["cluster_dilation_radius"]
    dil_fixed = morphology.dilation(nuclear_clean, morphology.disk(default_dr))
    lbl_fixed = measure.label(dil_fixed)
    lbl_nuc   = measure.label(nuclear_clean)
    nuc_props = measure.regionprops(lbl_nuc)
    cp_fixed  = measure.regionprops(lbl_fixed)
    cp_area   = {cp.label: cp.area for cp in cp_fixed}
    cnts_fixed: dict = {}
    for np_ in nuc_props:
        cy, cx = int(np_.centroid[0]), int(np_.centroid[1])
        cid = lbl_fixed[cy, cx]
        if cid > 0:
            cnts_fixed[cid] = cnts_fixed.get(cid, 0) + 1

    colors_mn = ["#ff9f43", "#48dbfb", "#ff6b9d", "#1dd1a1", "#a29bfe", "#ff6b6b"]
    pcts_mn = []
    for mn in min_nuclei_range:
        infl_px = sum(
            np_.area for np_ in nuc_props
            if (cid := lbl_fixed[int(np_.centroid[0]), int(np_.centroid[1])]) > 0
            and cnts_fixed.get(cid, 0) >= mn
            and cp_area.get(cid, 0) >= default_min_area
        )
        pcts_mn.append(100 * infl_px / tissue_px if tissue_px else 0)

    axes[1].plot(min_nuclei_range, pcts_mn, color="#06d6a0", linewidth=2.5, marker="o", markersize=5)
    axes[1].axvline(default_min_n, color="#ff6b6b", linestyle="--",
                    linewidth=1.5, label=f"Default ({default_min_n})")
    axes[1].set_xlabel("Min nuclei per qualifying cluster", fontsize=11)
    axes[1].set_ylabel("% Parenchymal inflammation (of tissue)", fontsize=11)
    axes[1].set_title(
        f"Sensitivity to min nuclei/cluster\n(dilation radius fixed at {default_dr} px)",
        color="white", fontsize=11
    )
    axes[1].legend(facecolor="#222", labelcolor="white", fontsize=9)

    plt.suptitle(
        f"Cluster Threshold Sensitivity — {result['file']}",
        color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# ── Group assignment ──────────────────────────────────────────────────────────

def assign_group(filename: str, group_prefixes: list[str]) -> str:
    """
    Assign an image to a group based on its filename prefix.
    Matching is case-insensitive. Returns 'Unknown' if no prefix matches.

    Example: "WT Aged1 Stitched 10x.tif" -> "WT"  (if "WT" is in group_prefixes)
    """
    name_upper = filename.upper()
    for prefix in group_prefixes:
        if name_upper.startswith(prefix.upper()):
            return prefix
    return "Unknown"


# ── Group comparison figure ───────────────────────────────────────────────────

def make_summary_figure(all_results: list[dict], out_path: Path, group_prefixes: list[str]) -> pd.DataFrame:
    """
    Publication-ready group comparison figure.

    Shows mean +/- SD bars with individual data points overlaid (sina/strip plot),
    plus Welch's t-test p-value annotation between the two primary groups.

    Returns a DataFrame of group statistics (also saved as group_stats.csv).
    """
    # ── Build per-group value lists ────────────────────────────────────────
    group_vals: dict[str, list[float]] = {}
    for r in all_results:
        g = r.get("group", "Unknown")
        group_vals.setdefault(g, []).append(r["infl_pct_of_tissue"])

    # Order groups: known prefixes first (in order given), then Unknown
    ordered_groups = [g for g in group_prefixes if g in group_vals]
    if "Unknown" in group_vals:
        ordered_groups.append("Unknown")

    # ── Compute stats ─────────────────────────────────────────────────────
    rows = []
    for g in ordered_groups:
        vals = np.array(group_vals[g])
        rows.append({
            "group":  g,
            "n":      len(vals),
            "mean":   float(np.mean(vals)),
            "sd":     float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "sem":    float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
        })
    stats_df = pd.DataFrame(rows)

    # ── Welch's t-test between first two groups ────────────────────────────
    p_value = None
    t_stat  = None
    if len(ordered_groups) >= 2:
        g1, g2 = ordered_groups[0], ordered_groups[1]
        v1, v2 = np.array(group_vals[g1]), np.array(group_vals[g2])
        if len(v1) >= 2 and len(v2) >= 2:
            t_stat, p_value = stats.ttest_ind(v1, v2, equal_var=False)
            stats_df.loc[stats_df["group"] == g1, "t_stat_vs_next"] = round(float(t_stat), 4)
            stats_df.loc[stats_df["group"] == g1, "p_value_vs_next"] = float(p_value)
            stats_df.loc[stats_df["group"] == g1, "comparison"]      = f"{g1} vs {g2} (Welch t-test)"

    # ── Figure ────────────────────────────────────────────────────────────
    PALETTE = ["#3a86ff", "#ff6b6b", "#06d6a0", "#ffd166", "#a29bfe"]
    fig, ax = plt.subplots(figsize=(max(6, len(ordered_groups) * 2.8), 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    x_positions = np.arange(len(ordered_groups))
    bar_width   = 0.55

    for i, g in enumerate(ordered_groups):
        vals  = np.array(group_vals[g])
        mean  = stats_df.loc[stats_df["group"] == g, "mean"].values[0]
        sd    = stats_df.loc[stats_df["group"] == g, "sd"].values[0]
        color = PALETTE[i % len(PALETTE)]

        # Bar
        ax.bar(x_positions[i], mean, width=bar_width, color=color,
               alpha=0.65, edgecolor="white", linewidth=0.8, zorder=2)

        # SD error bar
        ax.errorbar(x_positions[i], mean, yerr=sd,
                    fmt="none", color="white", linewidth=2, capsize=7, capthick=2, zorder=3)

        # Individual data points — jittered horizontally
        rng   = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_positions[i] + jitter, vals,
                   color="white", s=55, zorder=4, alpha=0.85, linewidths=0.5, edgecolors=color)

        # Mean label above bar
        ax.text(x_positions[i], mean + sd + (ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0.5),
                f"{mean:.2f}%", ha="center", va="bottom", color="white",
                fontsize=10, fontweight="bold")

    # ── Significance bracket ───────────────────────────────────────────────
    if p_value is not None and len(ordered_groups) >= 2:
        y_top   = max(
            stats_df["mean"].values[0] + stats_df["sd"].values[0],
            stats_df["mean"].values[1] + stats_df["sd"].values[1],
            max(group_vals[ordered_groups[0]]),
            max(group_vals[ordered_groups[1]]),
        )
        y_line  = y_top * 1.12
        y_text  = y_top * 1.16

        ax.plot([0, 0, 1, 1], [y_line * 0.97, y_line, y_line, y_line * 0.97],
                color="white", linewidth=1.5)

        if p_value < 0.0001:
            sig_str = "p < 0.0001 ****"
        elif p_value < 0.001:
            sig_str = f"p = {p_value:.4f} ***"
        elif p_value < 0.01:
            sig_str = f"p = {p_value:.4f} **"
        elif p_value < 0.05:
            sig_str = f"p = {p_value:.4f} *"
        else:
            sig_str = f"p = {p_value:.4f} (ns)"

        ax.text(0.5, y_text, sig_str,
                ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    # ── Axes formatting ────────────────────────────────────────────────────
    labels_with_n = [f"{g}\n(n={len(group_vals[g])})" for g in ordered_groups]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_with_n, color="white", fontsize=12)
    ax.set_ylabel("% Parenchymal inflammation (of tissue area)", color="white", fontsize=12)
    ax.set_title(
        "Parenchymal Inflammation — Group Comparison\n"
        "Bars: mean \u00b1 SD   \u25cf Individual animals   Welch\u2019s t-test",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    ax.set_xlim(-0.6, len(ordered_groups) - 0.4)
    y_max_data = max(v for vals in group_vals.values() for v in vals)
    ax.set_ylim(0, y_max_data * 1.35)
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()

    return stats_df


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantify parenchymal inflammation in H&E lung images (v3 — fixed threshold + group stats)."
    )
    parser.add_argument("--input",  "-i", required=True, help="Folder containing .tif images")
    parser.add_argument("--output", "-o", default="./results", help="Output folder (default: ./results)")
    parser.add_argument(
        "--reference", "-r",
        help=(
            "Path to a reference (WT/control) image with strong hematoxylin staining. "
            "The nuclear B-R threshold is calibrated from this image and applied uniformly. "
            "Required unless --nuclear-threshold is supplied. Quote paths with spaces."
        )
    )
    parser.add_argument(
        "--nuclear-threshold", "-t", type=float, default=None,
        help="Manually supply a fixed B-R nuclear threshold (e.g. 17.12). Overrides --reference."
    )
    parser.add_argument(
        "--groups", nargs="+", default=["WT", "FGF"],
        help=(
            "Filename prefixes used to assign images to groups (default: WT FGF). "
            "Images are matched case-insensitively from the start of the filename. "
            "Example: --groups Control Treated"
        )
    )
    parser.add_argument(
        "--extensions", nargs="+", default=[".tif", ".tiff", ".TIF", ".TIFF"],
        help="Image file extensions to process"
    )
    args = parser.parse_args()

    # ── Resolve nuclear threshold ──────────────────────────────────────────
    if args.nuclear_threshold is not None:
        nuclear_threshold = args.nuclear_threshold
        print(f"\nUsing manually supplied nuclear threshold: B-R > {nuclear_threshold:.2f}")
    elif args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"ERROR: Reference image not found: {ref_path}")
            sys.exit(1)
        print("\nCalibrating nuclear threshold from reference image...")
        nuclear_threshold = calibrate_nuclear_threshold(ref_path)
    else:
        print(
            "\nERROR: Supply either --reference <path/to/WT.tif> or --nuclear-threshold <value>.\n"
            "Example (Windows — note quotes around paths with spaces):\n"
            "  python lung_inflammation_analysis.py ^\n"
            '      --input "C:\\Images" ^\n'
            '      --output "C:\\Results" ^\n'
            '      --reference "C:\\Images\\WT Aged1 Stitched 10x.tif"\n'
        )
        sys.exit(1)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    val_dir    = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix in args.extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir} with extensions {args.extensions}")
        sys.exit(1)

    print(f"\nFound {len(image_files)} image(s) to process.")
    print(f"Group prefixes: {args.groups}\n")

    all_results = []

    for img_path in tqdm(image_files, desc="Analyzing"):
        try:
            result = analyze_image(img_path, nuclear_threshold=nuclear_threshold)

            # Assign group from filename prefix
            result["group"] = assign_group(img_path.name, args.groups)

            all_results.append(result)

            # Validation figure
            val_fig_path = val_dir / f"{img_path.stem}_validation.png"
            make_validation_figure(result, val_fig_path)

            # Threshold sensitivity figure
            sens_fig_path = val_dir / f"{img_path.stem}_threshold_sensitivity.png"
            make_threshold_sensitivity_figure(result, sens_fig_path)

            flag_str = "  ⚠ BIMODALITY WARN" if result["_bim_flag"] else ""
            tqdm.write(
                f"  ✓ [{result['group']:<8}] {img_path.name:<40}  "
                f"{result['infl_pct_of_tissue']:.2f}% infl  "
                f"bimodality={result['bimodality_score']:.3f}{flag_str}"
            )

        except Exception as e:
            tqdm.write(f"  ✗ {img_path.name} — ERROR: {e}")

    if not all_results:
        print("No results to save.")
        sys.exit(1)

    # ── Save per-image CSV ─────────────────────────────────────────────────
    csv_cols = [k for k in all_results[0].keys() if not k.startswith("_")]
    # Ensure 'group' appears early
    if "group" in csv_cols:
        csv_cols = ["file", "group"] + [c for c in csv_cols if c not in ("file", "group")]
    df = pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_results])
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # ── Group summary figure + stats CSV ─────────────────────────────────
    summary_path    = output_dir / "group_comparison.png"
    stats_df        = make_summary_figure(all_results, summary_path, group_prefixes=args.groups)
    stats_csv_path  = output_dir / "group_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'File':<42} {'Group':<8} {'Infl %':>7} {'Bimodality':>10}")
    print(f"{'─'*70}")
    for r in all_results:
        flag = " ⚠" if r["_bim_flag"] else "  "
        print(f"{r['file']:<42} {r['group']:<8} {r['infl_pct_of_tissue']:>6.2f}%  {r['bimodality_score']:>9.3f}{flag}")

    print(f"\n{'─'*70}")
    print(f"\nGROUP STATISTICS")
    print(f"{'─'*50}")
    for _, row in stats_df.iterrows():
        print(f"  {row['group']:<8}  n={int(row['n'])}   mean={row['mean']:.2f}%   SD={row['sd']:.2f}   SEM={row['sem']:.2f}")
    if "p_value_vs_next" in stats_df.columns:
        p_row = stats_df.dropna(subset=["p_value_vs_next"])
        if not p_row.empty:
            p = p_row.iloc[0]["p_value_vs_next"]
            comp = p_row.iloc[0]["comparison"]
            sig = "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"\n  Welch t-test ({comp}):")
            print(f"    p = {p:.6f}  {sig}")

    print(f"\n  Fixed nuclear threshold : B-R > {nuclear_threshold:.2f}")
    print(f"  (Reuse with: --nuclear-threshold {nuclear_threshold:.2f})")
    print(f"\nOutputs:")
    print(f"  Per-image results  : {csv_path}")
    print(f"  Group stats        : {stats_csv_path}")
    print(f"  Group figure       : {summary_path}")
    print(f"  Validation figures : {val_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
