"""
Export cropped cores for all rows in the paths table (Excel).
Uses the same cropping logic as export_core_nii.py; iterates over a dataframe
with columns: label ID, root_path, filename (filename may be NaN for list-of-DICOM-files mode).

Set paths_table and ini_root_path below, then run: python export_cores_batch.py
"""
import os
import pandas as pd

from utils import (
    get_paths,
    load_volume,
    load_labels,
    load_label_descriptions,
    resolve_label_index,
    crop_to_mask_region,
    save_cropped_volume_nii,
)

# --- Config ---
paths_table = "data_info/MNHT_paths_v1.xlsx"  # Excel with columns: label ID, root_path, filename
ini_root_path = ".."  # Base path for root_path column (as in notebook)
output_dir = "data/cores"  # Directory where each core .nii.gz is saved
min_extent = 10  # Minimum voxels per dimension (same as export_core_nii)
output_list_path = "data/cores_exported.csv"  # CSV of exported cores (label ID, root_path, filename, output_path); None to skip


def export_cores_from_table():
    df = pd.read_excel(paths_table)
    # Optional: drop rows with missing root_path (as in notebook)
    df = df.loc[~df["root_path"].isna()].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for index, row in df.iterrows():
        label_id = row["label ID"]
        root_path = row["root_path"]
        file_name = row["filename"]
        if pd.isna(file_name):
            file_name = None
        else:
            file_name = str(file_name).strip()

        base_path = ini_root_path
        full_root, dicom_path, labels_path, descriptions_path, _ = get_paths(
            root_path, file_name, base_path
        )

        if not os.path.isfile(labels_path):
            print(f"[{label_id}] Skip: labels not found {labels_path}")
            continue

        volume, _ = load_volume(root_path, file_name, base_path)
        if volume is None:
            print(f"[{label_id}] Skip: could not load volume from {full_root}")
            continue

        labels_dir = os.path.join(full_root, "labels")
        labels_img, labels_array = load_labels(labels_path, fallback_labels_dir=labels_dir)
        label_descriptions = load_label_descriptions(descriptions_path)

        try:
            label_index = resolve_label_index(label_id, label_descriptions)
        except ValueError as e:
            print(f"[{label_id}] Skip: {e}")
            continue

        try:
            cropped_volume, cropped_mask, new_affine = crop_to_mask_region(
                volume, labels_array, label_index, labels_img.affine, min_extent=min_extent
            )
        except ValueError as e:
            print(f"[{label_id}] Skip: {e}")
            continue

        safe_label = str(label_id).replace(" ", "_")
        out_path = os.path.join(output_dir, f"{safe_label}.nii.gz")
        save_cropped_volume_nii(cropped_volume, new_affine, out_path)
        results.append({
            "label ID": label_id,
            "root_path": root_path,
            "filename": file_name if file_name is not None else "",
            "output_path": os.path.normpath(out_path),
        })
        print(f"Saved {out_path}")

    if output_list_path and results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_list_path, index=False)
        print(f"Wrote list of {len(results)} cores to {output_list_path}")

    return results


if __name__ == "__main__":
    export_cores_from_table()
