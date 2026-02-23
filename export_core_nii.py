"""
Export a single core: load data, crop tight to the mask for the given labelID, save as nii.gz.

Set the variables below and run: python export_core_nii.py
labelID can be the integer index in the mask or the label name string (e.g. N1002).
"""
import os

from utils import (
    get_paths,
    load_volume,
    load_labels,
    load_label_descriptions,
    resolve_label_index,
    crop_to_mask_region,
    save_cropped_volume_nii,
)

# --- Input: set these directly ---
root_path = "..//MU//SOIL64_8_10_20"  # Root directory (relative to base_path or absolute)
file_name = "SOIL_FRIT_8_10_20Series_003_Bone 0.5 SCAN 1.2"  # Optional: single DICOM series name (no .dcm); None = load DICOMs from root_path
labelID = "N1002"  # Label index (int) or label name (e.g. N1002)
output = f"data/{labelID}.nii.gz"  # Output .nii.gz path; None = <root_basename>_<labelID>_cropped.nii.gz under root
base_path = "."  # Prefix for root_path
min_extent = 10  # Minimum voxels in each dimension (avoids tiny dimensions like 3); use 1 for tight crop only


def main():
    try:
        label_id_arg = int(labelID)
    except (ValueError, TypeError):
        label_id_arg = labelID

    full_root, dicom_path, labels_path, descriptions_path, _ = get_paths(
        root_path, file_name, base_path
    )

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    volume, _ = load_volume(root_path, file_name, base_path)
    if volume is None:
        raise FileNotFoundError(
            f"Could not load volume from {full_root} (file_name={file_name}). "
            "Check that DICOM files or the single .dcm file exist."
        )

    labels_dir = os.path.join(full_root, "labels")
    labels_img, labels_array = load_labels(labels_path, fallback_labels_dir=labels_dir)
    label_descriptions = load_label_descriptions(descriptions_path)

    label_index = resolve_label_index(label_id_arg, label_descriptions)

    labels_for_crop = labels_array
    cropped_volume, cropped_mask, new_affine = crop_to_mask_region(
        volume, labels_for_crop, label_index, labels_img.affine, min_extent=min_extent
    )

    if output:
        out_path = output
    else:
        base = os.path.basename(full_root.rstrip(os.sep))
        safe_label = str(label_id_arg).replace(" ", "_")
        out_path = os.path.join(full_root, f"{base}_{safe_label}_cropped.nii.gz")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_cropped_volume_nii(cropped_volume, new_affine, out_path)
    print(f"Saved cropped volume to {out_path}")


if __name__ == "__main__":
    main()
