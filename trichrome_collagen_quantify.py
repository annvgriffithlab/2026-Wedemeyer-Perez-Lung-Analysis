"""
trichrome_collagen_quantify.py
-------------------------------
Quantifies blue-staining (collagen) in Masson's Trichrome stained tissue images.

Usage:
    # Analyze all .tif files in a folder (recommended on Windows)
    python trichrome_collagen_quantify.py testcollagen/

    # Analyze specific files
    python trichrome_collagen_quantify.py image1.tif image2.tif

    # Wildcard (works on Mac/Linux; use a folder path on Windows instead)
    python trichrome_collagen_quantify.py testcollagen/*.tif

    # Or edit IMAGE_PATHS below and run without arguments.

Output:
    - Console table of results
    - <input_stem>_mask.png  — collagen mask overlay for each input image
    - collagen_comparison.png — bar chart comparing all samples

Dependencies:
    pip install Pillow numpy matplotlib
"""

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Configuration -------------------------------------------------------------

# Fallback image list if no CLI arguments are provided
IMAGE_PATHS = [
    "FGF_Aged1_Trichrome_Final_Stitch_10x.tif",
    "WT_Aged1_Trichrome_Final_Stitch_10x.tif",
]

# Background exclusion: pixels with mean brightness above this are background
BACKGROUND_BRIGHTNESS_THRESHOLD = 220  # 0-255

# Blue (collagen) detection criteria (all must be satisfied):
BLUE_OVER_RED_MARGIN   = 20  # B must exceed R by at least this much
BLUE_OVER_GREEN_MARGIN = 10  # B must exceed G by at least this much
BLUE_MIN_INTENSITY     = 80  # B must be above this (excludes very dark nuclei)

# Output filenames
COMPARISON_FIGURE = "collagen_comparison.png"

# -- Path resolution -----------------------------------------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def resolve_image_paths(raw_args):
    """
    Expand each argument into a list of image Paths.

    - If the argument is a directory, all image files inside are collected.
    - If the argument contains a wildcard (* or ?), glob is applied manually.
      This handles Windows CMD/PowerShell, which don't expand wildcards
      automatically the way Mac/Linux shells do.
    - Otherwise the argument is treated as a literal file path.
    """
    found = []

    for arg in raw_args:
        p = Path(arg)

        if p.is_dir():
            matches = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not matches:
                print(f"  WARNING: No image files found in folder '{p}'")
            found.extend(matches)

        elif any(c in arg for c in ("*", "?")):
            matches = sorted(Path(m) for m in glob.glob(arg))
            if not matches:
                print(f"  WARNING: No files matched pattern '{arg}'")
            found.extend(matches)

        else:
            found.append(p)

    return found


# -- Core analysis -------------------------------------------------------------

def quantify_collagen(image_path):
    """
    Load a Masson's Trichrome image and quantify blue (collagen) pixels.

    Returns a dict with:
        label       - filename stem
        path        - resolved Path
        image_array - uint8 RGB ndarray
        tissue_mask - bool mask (True = tissue)
        blue_mask   - bool mask (True = collagen)
        n_total     - total pixel count
        n_tissue    - tissue pixel count
        n_blue      - collagen pixel count
        pct_blue    - collagen as % of tissue
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Background: bright, near-white pixels
    brightness      = (R + G + B) / 3
    background_mask = brightness > BACKGROUND_BRIGHTNESS_THRESHOLD
    tissue_mask     = ~background_mask

    # Blue collagen: B dominant, not too dark
    blue_mask = (
        (B > R + BLUE_OVER_RED_MARGIN)   &
        (B > G + BLUE_OVER_GREEN_MARGIN) &
        (B > BLUE_MIN_INTENSITY)         &
        tissue_mask
    )

    n_total  = arr.shape[0] * arr.shape[1]
    n_tissue = int(tissue_mask.sum())
    n_blue   = int(blue_mask.sum())
    pct_blue = 100.0 * n_blue / n_tissue if n_tissue > 0 else 0.0

    return {
        "label":           path.stem,
        "path":            path,
        "image_array":     arr.astype(np.uint8),
        "tissue_mask":     tissue_mask,
        "background_mask": background_mask,
        "blue_mask":       blue_mask,
        "n_total":         n_total,
        "n_tissue":        n_tissue,
        "n_blue":          n_blue,
        "pct_blue":        pct_blue,
    }


# -- Figures -------------------------------------------------------------------

def save_mask_overlay(result, out_path):
    """Save a side-by-side original + collagen mask figure for one sample."""
    r = result
    mask_rgb = np.zeros((*r["blue_mask"].shape, 3), dtype=np.uint8)
    mask_rgb[r["tissue_mask"]]     = [220, 200, 200]  # pale pink = other tissue
    mask_rgb[r["blue_mask"]]       = [0,   100, 220]  # blue = collagen
    mask_rgb[r["background_mask"]] = [240, 240, 240]  # light grey = background

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r["label"], fontsize=13, fontweight="bold")

    axes[0].imshow(r["image_array"])
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title(
        f"Collagen Mask  --  {r['pct_blue']:.2f}% of tissue\n"
        "(blue = collagen, pink = other tissue, grey = background)",
        fontsize=11,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison_figure(results, out_path):
    """Bar chart comparing collagen % across all samples."""
    labels = [r["label"] for r in results]
    pcts   = [r["pct_blue"] for r in results]
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(results)), 5))
    bars = ax.bar(labels, pcts, color=colors, width=0.5, edgecolor="k", linewidth=0.8)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    ax.set_ylabel("Collagen (% of tissue area)", fontsize=12)
    ax.set_title("Masson's Trichrome -- Collagen Content Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(pcts) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=10, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantify collagen (blue staining) in Masson's Trichrome images."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=(
            "Image file(s), folder(s), or glob pattern(s). "
            "Examples:  testcollagen/   |   *.tif   |   img1.tif img2.tif. "
            "Falls back to IMAGE_PATHS at the top of the script if omitted."
        ),
    )
    args = parser.parse_args()

    if args.images:
        image_paths = resolve_image_paths(args.images)
    else:
        image_paths = [Path(p) for p in IMAGE_PATHS]

    if not image_paths:
        parser.error("No images found. Pass a folder, wildcard, or file path(s).")

    print(f"Found {len(image_paths)} image(s) to process.\n")

    results = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            r = quantify_collagen(img_path)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            continue

        results.append(r)

        out_mask = Path(r["label"] + "_mask.png")
        save_mask_overlay(r, out_mask)
        print(f"  Collagen: {r['pct_blue']:.2f}% of tissue  |  mask saved -> {out_mask}")

    if not results:
        print("\nNo images were processed successfully. Exiting.")
        sys.exit(1)

    # Summary table
    print()
    print("=" * 62)
    print(f"{'Sample':<35} {'Tissue px':>10} {'Collagen px':>12} {'%':>4}")
    print("-" * 62)
    for r in results:
        print(f"{r['label']:<35} {r['n_tissue']:>10,} {r['n_blue']:>12,} {r['pct_blue']:>3.2f}%")
    print("=" * 62)

    if len(results) >= 2:
        out_cmp = Path(COMPARISON_FIGURE)
        save_comparison_figure(results, out_cmp)
        print(f"\nComparison figure saved -> {out_cmp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
