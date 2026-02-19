"""
Shared utilities for loading DICOM/labels and cropping to mask regions.
Used by visualization.py and export_core_nii.py.
"""
import os
import numpy as np
import pydicom
import nibabel as nib


def get_paths(root_path, file_name=None, base_path="."):
    """
    Resolve paths for DICOM, labels, and descriptions.

    Args:
        root_path: Root directory (relative to base_path or absolute).
        file_name: Optional. If given, use single-DICOM naming; else use folder-based naming.
        base_path: Prefix for root_path (default ".").

    Returns:
        full_root: Resolved root directory path.
        dicom_path: Path to single .dcm file if file_name is set, else None.
        labels_path: Path to corelabels .nii.gz.
        descriptions_path: Path to descriptions .txt.
        single_file_mode: True if file_name was provided (dicom_path may still be None if missing).
    """
    full_root = os.path.normpath(os.path.join(base_path, root_path)) if base_path else os.path.normpath(root_path)
    single_file_mode = file_name is not None

    if file_name is not None:
        dicom_path = os.path.join(full_root, file_name + ".dcm")
        labels_path = os.path.join(full_root, "labels", file_name + "_corelabels" + ".nii.gz")
        descriptions_path = os.path.join(full_root, "labels", file_name + "_descriptions" + ".txt")
    else:
        dicom_path = None
        labels_dir = os.path.join(full_root, "labels")
        prefix = os.path.basename(full_root) + "_"
        labels_path = os.path.join(labels_dir, prefix + "corelabels.nii.gz")
        descriptions_path = os.path.join(labels_dir, prefix + "descriptions.txt")

    return full_root, dicom_path, labels_path, descriptions_path, single_file_mode


def load_volume_from_dicom_files(root, exclude=("labels",)):
    """
    Load 3D volume when root contains DICOM files directly (e.g. 00000, 00001, 00010, ...).
    Returns array shape (Z, Y, X) or None if no valid slices.
    """
    if not os.path.isdir(root):
        return None
    try:
        entries = [
            name for name in os.listdir(root)
            if name not in exclude and os.path.isfile(os.path.join(root, name)) and not name.startswith(".")
        ]
    except OSError:
        return None
    if not entries:
        return None

    def sort_key(name):
        try:
            return (0, int(name.strip()))
        except ValueError:
            return (1, name)

    ordered = sorted(entries, key=sort_key)
    slices = []
    target_shape = None
    for name in ordered:
        path = os.path.join(root, name)
        try:
            ds = pydicom.dcmread(path)
            arr = np.squeeze(ds.pixel_array)
        except Exception:
            continue
        if target_shape is None:
            target_shape = arr.shape
        if arr.shape == target_shape:
            slices.append(arr)
    if not slices:
        return None
    return np.stack(slices, axis=0)


def load_volume(root_path, file_name=None, base_path="."):
    """
    Load 3D volume (Z, Y, X) from either a single DICOM file or a directory of DICOM files.

    Returns:
        volume: np.ndarray shape (Z, Y, X), or None if loading failed.
        from_dicom_list: True if volume was built from multiple DICOM files in root (for grayscale display).
    """
    _, dicom_path, _, _, single_file_mode = get_paths(root_path, file_name, base_path)
    full_root = os.path.normpath(os.path.join(base_path, root_path)) if base_path else os.path.normpath(root_path)

    if single_file_mode and dicom_path is not None and os.path.isfile(dicom_path):
        ds = pydicom.dcmread(dicom_path)
        pixel_array = np.squeeze(ds.pixel_array)
        if pixel_array.ndim == 2:
            folder = os.path.dirname(dicom_path)
            all_dcm = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
            if len(all_dcm) > 1:
                target_shape = pixel_array.shape
                loaded = [pixel_array]
                for f in sorted(all_dcm):
                    if f == os.path.basename(dicom_path):
                        continue
                    p = os.path.join(folder, f)
                    try:
                        a = np.squeeze(pydicom.dcmread(p).pixel_array)
                        if a.shape == target_shape:
                            loaded.append(a)
                    except Exception:
                        continue
                if len(loaded) > 1:
                    pixel_array = np.stack(loaded, axis=0)
        if pixel_array.ndim == 2:
            pixel_array = np.expand_dims(pixel_array, axis=0)  # (Y, X) -> (1, Y, X) for consistent (Z, Y, X)
        return pixel_array, False
    else:
        pixel_array = load_volume_from_dicom_files(full_root, exclude=("labels",))
        if pixel_array is None:
            return None, False
        if pixel_array.ndim == 2:
            pixel_array = np.expand_dims(pixel_array, axis=0)
        return pixel_array, True


def load_label_descriptions(descriptions_path):
    """
    Load ITK-SNAP format descriptions file: IDX  R G B A  VIS MSH  "LABEL".
    Returns dict mapping integer index -> label name string.
    """
    label_descriptions = {}
    if not os.path.isfile(descriptions_path):
        return label_descriptions
    with open(descriptions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                idx = int(parts[0])
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and end > start:
                    label_descriptions[idx] = line[start + 1 : end]
                else:
                    label_descriptions[idx] = parts[-1]
            except (ValueError, IndexError):
                pass
    return label_descriptions


def load_labels(labels_path, fallback_labels_dir=None):
    """
    Load labels NIfTI. Tries labels_path, then fallback_labels_dir/corelabels.nii.gz and descriptions if given.

    Returns:
        labels_img: nibabel NIfTI image (for affine).
        labels_array: np.ndarray (X, Y, Z) int32, squeezed.
    """
    if not os.path.isfile(labels_path) and fallback_labels_dir:
        labels_path = os.path.join(fallback_labels_dir, "corelabels.nii.gz")
    labels_img = nib.load(labels_path)
    labels_array = np.asarray(labels_img.dataobj).astype(np.int32)
    labels_array = np.squeeze(labels_array)
    return labels_img, labels_array


def resolve_label_index(label_id, label_descriptions):
    """
    Resolve label_id to the integer index used in the mask.
    label_id can be an int (used directly) or a string (matched to label_descriptions value).
    """
    if isinstance(label_id, (int, np.integer)):
        return int(label_id)
    for idx, name in label_descriptions.items():
        if name == label_id:
            return idx
    raise ValueError(f"Label ID '{label_id}' not found in descriptions. Known: {list(label_descriptions.values())}")


def crop_to_mask_region(volume, labels_array, label_index, labels_affine):
    """
    Crop volume and mask to the tight bounding box of the region where mask == label_index.

    volume: (Z, Y, X) - intensity volume
    labels_array: (X, Y, Z) - label map from NIfTI
    label_index: int - label value to crop to
    labels_affine: 4x4 affine from the labels NIfTI (voxel indices in X, Y, Z)

    Returns:
        cropped_volume: (Z, Y, X) cropped intensity
        cropped_mask: (X, Y, Z) cropped binary mask (0/1) for label_index
        new_affine: 4x4 affine for the cropped image (so cropped can be saved as NIfTI in same physical space)
    """
    mask = (labels_array == label_index)
    if not np.any(mask):
        raise ValueError(f"No voxels found for label index {label_index}")

    # labels_array is (X, Y, Z)
    xs, ys, zs = np.where(mask)
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    z_min, z_max = int(np.min(zs)), int(np.max(zs))

    cropped_labels = labels_array[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    cropped_mask = (cropped_labels == label_index).astype(np.int32)

    # volume is (Z, Y, X)
    cropped_volume = volume[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1].copy()

    # New affine: voxel (0,0,0) in cropped image = (x_min, y_min, z_min) in original
    new_affine = np.eye(4)
    new_affine[:3, :3] = labels_affine[:3, :3]
    new_affine[:3, 3] = (labels_affine @ np.array([x_min, y_min, z_min, 1.0]))[:3]

    return cropped_volume, cropped_mask, new_affine


def save_cropped_volume_nii(cropped_volume, new_affine, out_path):
    """
    Save cropped volume (Z, Y, X) as NIfTI .nii.gz with the given affine.
    Stored in file as (X, Y, Z) to match NIfTI convention.
    Handles 2D (adds slice dim) and 4D+ (squeezes singleton dims) so result is 3D.
    """
    vol = np.squeeze(cropped_volume)
    if vol.ndim == 2:
        vol = np.expand_dims(vol, axis=0)  # (Y, X) -> (1, Y, X)
    elif vol.ndim > 3:
        # Keep only first 3 dimensions (e.g. (1, Z, Y, X) -> take [0])
        while vol.ndim > 3:
            vol = vol[(0,) * (vol.ndim - 3)]
    # NIfTI convention: (X, Y, Z)
    vol_xzy = np.transpose(vol, (2, 1, 0))
    img = nib.Nifti1Image(vol_xzy.astype(np.float32), new_affine)
    nib.save(img, out_path)
