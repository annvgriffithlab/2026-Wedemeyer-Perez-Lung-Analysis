# Image Processing Pipelines for Microscopy & Histology

This repository contains the Python scripts used for automated quantitative histology analysis in our manuscript. It features pipelines for brightfield color segmentation (Masson's Trichrome) and cluster-based cellular infiltration analysis (H&E).

## Manuscript Citation
> **[Sarah A. Wedemeyer1, Jaid Perez1, Kahealani S. Archuleta1, Danny Bui1, Tariq Elhashim1, Emma C. Collins1, Courtney C. Segura-Cepero1, Yangming Xiao1, Michael J. Wedemeyer3, Nu Zhang1,2, Elizabeth A. Leadbetter1, Ann V. Griffith<img width="468" height="34" alt="image" src="https://github.com/user-attachments/assets/8620c66a-856b-4005-be1f-4092c6cf427d" />
].** (2026). *[Sustained thymic stromal FGF21 expression improves peripheral CD4+ and CD+ T cell anti-viral and anti-tumor effector responses during aging]*. (Submitted for Publication).

---

## 🛠️ Combined Requirements & Installation

The scripts utilize standard scientific Python libraries along with advanced computer vision and image processing toolkits. Ensure your environment has the required dependencies installed:

```bash
pip install numpy scipy scikit-image pillow matplotlib opencv-python pandas tqdm
```
*Note: These scripts are verified and stable on Python 3.12+.*

---

## 🔬 1. Masson's Trichrome Collagen Quantification
**File:** `trichrome_collagen_quantify.py`

This script automates the quantification of blue-staining (collagen) area in Masson's Trichrome stained tissue sections by applying color-dominance thresholding.

### How it Works
1. **Background Exclusion:** Automatically isolates and discards bright brightfield backgrounds.
2. **Color Segmentation:** Identifies collagen pixels by enforcing a dominant blue channel rule (`B > R + margin` and `B > G + margin`) while excluding dark nuclear structures.
3. **Quantification:** Calculates the final collagen area as a strict percentage of total tissue area.

### Usage Examples
```bash
# Option A: Process an entire folder of images (Recommended on Windows)
python trichrome_collagen_quantify.py testcollagen/

# Option B: Process explicit image file pathways
python trichrome_collagen_quantify.py image1.tif image2.tif

# Option C: Use wildcards (Mac / Linux)
python trichrome_collagen_quantify.py testcollagen/*.tif
```
*Alternative: You can open the file and directly update the `IMAGE_PATHS` list to run the script without any terminal arguments.*

### Outputs
* **`[sample_name]_mask.png`**: A high-resolution side-by-side visual validation image. The mask displays background (grey), general tissue (pink), and isolated collagen (blue).
* **`collagen_comparison.png`**: A bar chart plotting and comparing the collagen content percentages across all processed samples.
* **Console Table**: A comprehensive terminal printout tracking raw pixel counts and final percentages.

---

## 🫁 2. Lung Parenchymal Inflammation Quantification (H&E)
**File:** `lung_inflammation_analysis.py`

An advanced cluster-based cell profiling pipeline designed to segment individual hematoxylin-stained nuclei, detect localized structural groups, and quantify inflammatory cell infiltration area in H&E lung histology sections.

### Critical Methodological Design (Fixed Threshold Calibration)
To overcome the limitations of per-image Otsu thresholding—which fails when experimental groups show reduced staining intensity—this script **calibrates a fixed nuclear threshold from a designated control/WT reference image**. This fixed criteria is then uniformly applied across your entire batch to guarantee identical detection sensitivity. The script also measures a **Bimodality QC Score** for every image to flag samples with poor staining contrast.

### Usage Examples
*Always wrap file paths containing spaces inside quotation marks.*

**Windows Command Prompt:**
```cmd
python lung_inflammation_analysis.py ^
    --input "C:\Images" ^
    --output "C:\Results" ^
    --reference "C:\Images\WT_Reference_10x.tif"
```

**Mac / Linux Terminal:**
```bash
python lung_inflammation_analysis.py \
    --input "/data/images" \
    --output "/data/results" \
    --reference "/data/images/WT_Reference_10x.tif"
```

**Re-run using a previously calculated threshold (Skips calibration):**
```bash
python lung_inflammation_analysis.py --input ./images --output ./results --nuclear-threshold 17.12
```

### Outputs
All tabular files and figures are exported directly to your designated `--output` directory:
* **`results.csv`**: Comprehensive per-image metrics, including total tissue/airspace percentages, nuclear counts, cluster counts, and Bimodality QC status.
* **`group_stats.csv`**: Consolidated group metrics reporting group sizes (\(n\)), means, Standard Deviations (SD), and Standard Errors of the Mean (SEM).
* **`group_comparison.png`**: A publication-ready figure featuring mean \(\pm\) SD bar charts overlaid with jittered individual data points and an annotated **Welch's t-test p-value**.
* **`validation/` Directory**: Contains two critical validation plots for every single processed image:
  1. `*_validation.png`: A 6-panel summary mapping out tissue masking, individual cell detection, candidate clusters, final inflammatory masks, and complete statistics for easy principal investigator review.
  2. `*_threshold_sensitivity.png`: Parameter sensitivity plots charting how your final outcomes change when adjusting dilation or cluster sizes, reinforcing structural integrity.

---

## 📄 License
This repository is open-source and licensed under the **MIT License**. See the `LICENSE` file for details.
