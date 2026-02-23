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


def _ensure_volume_4d_if_channels(volume):
    """
    If volume is 3D with one dimension of size 3 (RGB channels), reshape to 4D (Z, Y, X, C).
    volume: (Z, Y, X). Returns (Z, Y, X, C) with C=3 when a channel dimension is detected; otherwise unchanged.
    """
    if volume.ndim != 3:
        return volume
    sh = volume.shape
    channel_axis = None
    if sh[-1] == 3 and sh[0] > 3 and sh[1] > 3:
        channel_axis = 2
    elif sh[0] == 3 and sh[1] > 3 and sh[2] > 3:
        channel_axis = 0
    elif sh[1] == 3 and sh[0] > 3 and sh[2] > 3:
        channel_axis = 1
    if channel_axis is None:
        return volume
    # Move channel dimension to last: (Z, Y, X, C)
    perm = [i for i in range(3) if i != channel_axis] + [channel_axis]
    out = np.transpose(volume, perm)
    if channel_axis == 0:
        # (C, Y, X) -> (1, Y, X, C)
        out = np.expand_dims(out, axis=0)
    elif channel_axis == 2:
        # (Z, Y, C) -> (Z, Y, 1, C)
        out = np.expand_dims(out, axis=2)
    # channel_axis == 1: (Z, C, X) -> (Z, 1, X, C)
    else:
        out = np.expand_dims(out, axis=1)
    return out


def load_volume(root_path, file_name=None, base_path="."):
    """
    Load volume from either a single DICOM file or a directory of DICOM files.

    Returns:
        volume: np.ndarray shape (Z, Y, X) or (Z, Y, X, C) when a channel dimension exists (e.g. RGB).
                (Z, Y, X, C) has four dimensions with C=3 for color.
        from_dicom_list: True if volume was built from multiple DICOM files in root.
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
            pixel_array = np.expand_dims(pixel_array, axis=0)  # (Y, X) -> (1, Y, X)
        pixel_array = _ensure_volume_4d_if_channels(pixel_array)
        if pixel_array.ndim == 2:
            pixel_array = np.expand_dims(pixel_array, axis=0)
        return pixel_array, False
    else:
        pixel_array = load_volume_from_dicom_files(full_root, exclude=("labels",))
        if pixel_array is None:
            return None, False
        if pixel_array.ndim == 2:
            pixel_array = np.expand_dims(pixel_array, axis=0)
        pixel_array = _ensure_volume_4d_if_channels(pixel_array)
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


def _labels_to_volume_axes(labels_shape, volume_shape):
    """
    Determine how labels array axes map to volume (Z, Y, X).
    volume_shape may be 3D (Z, Y, X) or 4D (Z, Y, X, C); spatial part is used.
    Returns (z_ax, y_ax, x_ax): labels axis indices that correspond to volume Z, Y, X.
    """
    L0, L1, L2 = labels_shape[0], labels_shape[1], labels_shape[2]
    # Use spatial shape only (first 3 dims) for 4D volumes
    vol_spatial = volume_shape[:3]
    Vz, Vy, Vx = vol_spatial[0], vol_spatial[1], vol_spatial[2]
    # Slice dimension (Z) usually matches volume.shape[0].
    if L0 == Vz and L1 == Vy and L2 == Vx:
        return (0, 1, 2)  # labels are (Z, Y, X)
    if L2 == Vz and L1 == Vy and L0 == Vx:
        return (2, 1, 0)  # labels are (X, Y, Z)
    # Fallback: assume slice dimension is the one that matches Vz
    if L0 == Vz:
        return (0, 1, 2)
    if L2 == Vz:
        return (2, 1, 0)
    if L1 == Vz:
        return (1, 0, 2)  # (Y, Z, X) or similar
    return (2, 1, 0)  # default (X, Y, Z)


def crop_to_mask_region(volume, labels_array, label_index, labels_affine, min_extent=1):
    """
    Crop volume and mask to the tight bounding box of the region where mask == label_index.

    volume: (Z, Y, X) or (Z, Y, X, C) - intensity volume (4D when color channels present)
    labels_array: label map from NIfTI, either (X, Y, Z) or (Z, Y, X) depending on how it was saved
    label_index: int - label value to crop to
    labels_affine: 4x4 affine from the labels NIfTI (voxel indices follow labels array order)
    min_extent: int - minimum size in each dimension (default 1). Use e.g. 10 so no dimension is tiny.

    Returns:
        cropped_volume: (Z, Y, X) or (Z, Y, X, C) cropped intensity
        cropped_mask: (X, Y, Z) cropped binary mask (0/1) for label_index
        new_affine: 4x4 affine for the cropped image
    """
    mask = (labels_array == label_index)
    if not np.any(mask):
        raise ValueError(f"No voxels found for label index {label_index}")

    vol_spatial = volume.shape[:3]
    z_ax, y_ax, x_ax = _labels_to_volume_axes(labels_array.shape, volume.shape)
    all_inds = np.where(mask)
    a0_min, a0_max = int(np.min(all_inds[z_ax])), int(np.max(all_inds[z_ax]))
    a1_min, a1_max = int(np.min(all_inds[y_ax])), int(np.max(all_inds[y_ax]))
    a2_min, a2_max = int(np.min(all_inds[x_ax])), int(np.max(all_inds[x_ax]))

    z_min, z_max = a0_min, a0_max
    y_min, y_max = a1_min, a1_max
    x_min, x_max = a2_min, a2_max

    # Enforce minimum extent in each dimension (pad around center, clamp to volume bounds)
    if min_extent > 1:
        Vz, Vy, Vx = vol_spatial[0], vol_spatial[1], vol_spatial[2]

        def pad_clamp(lo, hi, size):
            extent = hi - lo + 1
            if extent < min_extent:
                need = min_extent - extent
                lo = max(0, lo - (need + 1) // 2)
                hi = min(size - 1, hi + need // 2)
                lo = max(0, min(lo, hi - min_extent + 1))
                hi = min(size - 1, max(hi, lo + min_extent - 1))
            return lo, hi

        z_min, z_max = pad_clamp(z_min, z_max, Vz)
        y_min, y_max = pad_clamp(y_min, y_max, Vy)
        x_min, x_max = pad_clamp(x_min, x_max, Vx)

    # Cropped labels for mask (bbox in label array axis order)
    idx = [slice(None)] * 3
    idx[z_ax], idx[y_ax], idx[x_ax] = slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1)
    cropped_labels = labels_array[tuple(idx)]
    cropped_mask = (cropped_labels == label_index).astype(np.int32)

    # volume is (Z, Y, X) or (Z, Y, X, C)
    cropped_volume = volume[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1].copy()

    # New affine: cropped voxel (0,0,0) corresponds to label voxel (z_min, y_min, x_min) in (Z,Y,X)
    orig_in_label_order = np.zeros(4)
    orig_in_label_order[z_ax], orig_in_label_order[y_ax], orig_in_label_order[x_ax] = z_min, y_min, x_min
    orig_in_label_order[3] = 1.0
    new_affine = np.eye(4)
    new_affine[:3, :3] = labels_affine[:3, :3]
    new_affine[:3, 3] = (labels_affine @ orig_in_label_order)[:3]

    return cropped_volume, cropped_mask, new_affine


def save_cropped_volume_nii(cropped_volume, new_affine, out_path):
    """
    Save cropped volume as NIfTI .nii.gz with the given affine.
    Accepts (Z, Y, X) or (Z, Y, X, C). If 4D (channels), converts to grayscale (luminance) before saving.
    Stored as (X, Y, Z) so the slice dimension (Z) is last - matches common
    viewers (e.g. ITK-SNAP, reference C1216.nii.gz with shape (209, 171, 1006)).
    """
    vol = np.asarray(cropped_volume)
    if vol.ndim == 4:
        # (Z, Y, X, C) -> grayscale (Z, Y, X) via luminance
        vol = np.tensordot(vol, [0.299, 0.587, 0.114], axes=(-1, 0)).astype(np.float32)
    vol = np.squeeze(vol)
    if vol.ndim == 2:
        vol = np.expand_dims(vol, axis=0)  # (Y, X) -> (1, Y, X)
    elif vol.ndim > 3:
        vol = vol[(0,) * (vol.ndim - 3)]
    # Store as (X, Y, Z) so slice dimension is last
    vol_xyz = np.transpose(vol, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    aff = np.eye(4)
    aff[:3, 0] = new_affine[:3, 2]   # first voxel axis = x
    aff[:3, 1] = new_affine[:3, 1]   # second = y
    aff[:3, 2] = new_affine[:3, 0]   # third = z (slice)
    aff[:3, 3] = new_affine[:3, 3]   # translation unchanged
    img = nib.Nifti1Image(vol_xyz.astype(np.float32), aff)
    nib.save(img, out_path)
