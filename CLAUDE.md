# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project evaluates deep learning models for predicting **root morphology features** from CT scan data of soil cores (University of Missouri). Four segmentation models are compared — DynUNet, SegResNet, UNet, UNETR — against expert-annotated ground truth measurements.

**Key morphology features**: Root Length (cm), Projected Area (cm²), Surface Area (cm²), Average Diameter (mm), Root Length by diameter class (L00–L40: five 1 mm-wide diameter bins, last bin > 4 mm).

## Environment Setup

```bash
pip install -r requirements.txt
```

Dependencies: `pydicom`, `nibabel`, `numpy`, `matplotlib`.

## Running Scripts

Scripts are run directly — no build step, no Makefile.

```bash
# Compute model performance metrics (primary evaluation script)
python ComputeMetrics.py

# Export a single soil core from DICOM → NIfTI
# (Edit root_path, file_name, labelID variables first)
python export_core_nii.py

# Batch export all cores listed in MNHT_paths_v1.xlsx
python export_cores_batch.py

# Interactive DICOM volume viewer with slice slider
python visualization.py

# Feature scatter plots: predictions vs ground truth
python visualizeANDcalibrate.py

# Compare previous vs current model results side-by-side
python compare_prev_cur.py

# Launch notebooks
jupyter notebook descriptive_analysis.ipynb
jupyter notebook notebook_performance.ipynb
```

## Architecture

### Data Pipeline

```
DICOM files
  → utils.load_volume() + crop_to_mask_region()
  → data/cores/*.nii.gz  (one file per soil core)
  → deep learning model inference (external)
  → data_info/prediction_{model}_{split}.csv
  → ComputeMetrics.py
  → data_info/model_metrics_{split}_feat_gt_Length(cm).csv
```

### Key Files

| File | Role |
|------|------|
| `utils.py` | Shared I/O: load DICOM/NIfTI volumes, parse ITK-SNAP labels, crop to bounding box, save NIfTI |
| `ComputeMetrics.py` | Main evaluation: computes Pearson r, Spearman ρ, R², MSE, mean/std error for each model |
| `data_info/MNHT_rootmorphology_GT_v2.csv` | Ground truth morphology features |
| `data_info/prediction_{model}_{split}.csv` | Per-model predictions (dynunet, segresnet, unet, unetr × test/val) |
| `data_info/CoresGT_val.csv` | Validation split ground truth |
| `data/cores_exported.csv` | Index mapping label IDs to exported NIfTI paths |

### `utils.py` Key Functions

- `get_paths(root_path, file_name)` — resolves DICOM/label file locations
- `load_volume(path)` — loads single DICOM or directory of DICOMs; handles 3D (Z,Y,X) and 4D (Z,Y,X,C) including RGB
- `load_labels(path)` — loads NIfTI segmentation mask
- `load_label_descriptions(path)` — parses ITK-SNAP label description format
- `crop_to_mask_region(volume, labels, labelID, min_extent)` — crops volume to tight bounding box of a specific label
- `save_cropped_volume_nii(volume, affine, output_path)` — writes NIfTI file preserving affine transform

### Data Conventions

- NIfTI volumes use axis order **(Z, Y, X)**; scripts are axis-mapping aware
- `ComputeMetrics.py` standardizes features on the full matched split, then evaluates on a selected subset
- `.gitignore` excludes `*.nii.gz` and Excel lock files — large binary data is not tracked in git
- Old/archived predictions live in `data_info/old_results/` and `data_info/results_v1/`
