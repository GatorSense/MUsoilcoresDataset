import os
import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider

# Volumes: (1) single DICOM file (set file_name) or (2) list of DICOM files in root_path (file_name = None).
root_path = r"..\MU\1.2.392.200036.9116.2.6.1.48.1215780058.1595283802.490100\1.2.392.200036.9116.2.6.1.48.1215780058.1595286344.274290"
file_name = None  # None = list of DICOM files in root (e.g. 00000, 00010); else single .dcm file name

# root_path = r"..\MU\SOIL_8_10_20_4"
# file_name = "SOIL_8_10_20_4Series_003_Bone 0.5 SCAN 10"

if file_name is not None:
    dicom_path = os.path.join(root_path, file_name + ".dcm")
    labels_path = os.path.join(root_path, "labels", file_name + "_corelabels" + ".nii.gz")
    descriptions_path = os.path.join(root_path, "labels", file_name + "_descriptions" + ".txt")
else:
    dicom_path = None
    labels_dir = os.path.join(root_path, "labels")
    prefix = os.path.basename(root_path) + "_"  # e.g. 002_corelabels.nii.gz
    labels_path = os.path.join(labels_dir, prefix + "corelabels.nii.gz")
    descriptions_path = os.path.join(labels_dir, prefix + "descriptions.txt")


def load_volume_from_dicom_files(root, exclude=("labels",)):
    """Load 3D volume when root contains DICOM files directly (e.g. 00000, 00001, 00010, ...)."""
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
    # Sort by name (numeric if names are integers, else lexicographic)
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


# Load raw data: single DICOM file or list of DICOM files in root
if dicom_path is not None and os.path.isfile(dicom_path):
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
                a = np.squeeze(pydicom.dcmread(p).pixel_array)
                if a.shape == target_shape:
                    loaded.append(a)
            if len(loaded) > 1:
                pixel_array = np.stack(loaded, axis=0)
else:
    # Volume from list of DICOM files in root (e.g. 00000, 00001, 00010, ...)
    dicom_path = None
    pixel_array = load_volume_from_dicom_files(root_path, exclude=("labels",))
    if pixel_array is None:
        path_err = ""
        try:
            raw_list = os.listdir(root_path) if os.path.isdir(root_path) else []
        except OSError as e:
            raw_list = []
            path_err = f" (os error: {e})"
        files_in_root = [x for x in raw_list if os.path.isfile(os.path.join(root_path, x))]
        hint = f" Files in root: {files_in_root[:10]}{'...' if len(files_in_root) > 10 else ''}." if files_in_root else f" Directory listing: {raw_list[:15]}."
        raise FileNotFoundError(
            f"No volume found under {root_path}.{path_err} "
            f"Expected DICOM files in root (e.g. 00000, 00010).{hint}"
        )

# Load labels (NIfTI from ITK-SNAP)
if not os.path.isfile(labels_path):
    labels_path = os.path.join(root_path, "labels", "corelabels.nii.gz")
    descriptions_path = os.path.join(root_path, "labels", "descriptions.txt")
labels_img = nib.load(labels_path)
labels_array = np.asarray(labels_img.dataobj).astype(np.int32)
labels_array = np.squeeze(labels_array)

# Load descriptions (ITK-SNAP format: IDX  R G B A  VIS MSH  "LABEL")
label_descriptions = {}
if os.path.isfile(descriptions_path):
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
                # Label is the last quoted string
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and end > start:
                    label_descriptions[idx] = line[start + 1 : end]
                else:
                    label_descriptions[idx] = parts[-1]
            except (ValueError, IndexError):
                pass

# Align labels to pixel_array: (512, 512, 2221) -> (2221, 512, 512)
labels_aligned = np.transpose(labels_array, (2, 1, 0))

# Colormap for labels (0 = transparent)
n_labels = int(np.max(labels_array)) + 1
_cmap = plt.colormaps["nipy_spectral"].resampled(max(n_labels, 2))
colors = np.array(_cmap(np.arange(_cmap.N)))
colors[0, 3] = 0
label_cmap = ListedColormap(colors)

n_slices = pixel_array.shape[0]
rgb_max = pixel_array.max() if pixel_array.max() > 1 else 1
# Grayscale for scan when volume was loaded from list of DICOM files
volume_from_list_of_files = dicom_path is None


def overlay_volume(alpha=0.5):
    """Interactive viewer: slider to move through slices."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.12)
    slice_idx = n_slices // 2
    rgb = pixel_array[slice_idx] / rgb_max
    lab = labels_aligned[slice_idx]
    scan_cmap = "gray" if volume_from_list_of_files else None
    im_rgb = ax.imshow(rgb, cmap=scan_cmap)
    masked = np.ma.masked_where(lab == 0, lab)
    im_lab = ax.imshow(masked, cmap=label_cmap, alpha=alpha, vmin=0, vmax=n_labels)
    ax.set_title(f"Slice {slice_idx} / {n_slices - 1}")
    ax.axis("off")

    # Legend from descriptions_path
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=label_descriptions.get(i, str(i)))
        for i in sorted(label_descriptions.keys())
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(slider_ax, "Slice", 0, n_slices - 1, valinit=slice_idx, valstep=1)

    def update(val):
        i = int(slider.val)
        ax.set_title(f"Slice {i} / {n_slices - 1}")
        im_rgb.set_data(pixel_array[i] / rgb_max)
        im_lab.set_data(np.ma.masked_where(labels_aligned[i] == 0, labels_aligned[i]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    overlay_volume()
