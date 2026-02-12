import os
import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider

# Folder that contains both slice subfolders (00000, 00001, ...) and a "labels" subfolder
root_path = r"D:\jose_data\MU\DICOM_03\0000\002"
file_name = None  # None = volume is a list of files (numbered subfolders); else single file name

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


def load_volume_from_numbered_folders(root, exclude=("labels",)):
    """Load 3D volume when root contains subfolders 00000, 00001, ... (one image per folder)."""
    if not os.path.isdir(root):
        return None
    try:
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in exclude]
    except OSError:
        return None
    # Prefer folders whose name parses as int; otherwise use all subdirs sorted by name
    numeric = []
    for d in subdirs:
        try:
            int(d.strip())
            numeric.append(d)
        except ValueError:
            pass
    if numeric:
        ordered = sorted(numeric, key=lambda x: int(x.strip()))
    else:
        ordered = sorted(subdirs)
    if not ordered:
        return None
    slices = []
    target_shape = None
    known_img = (".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    for name in ordered:
        folder = os.path.join(root, name)
        try:
            all_files = [f for f in os.listdir(folder) if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))]
        except OSError:
            continue
        files = [f for f in all_files if f.lower().endswith(known_img)]
        if not files:
            files = all_files  # include extensionless files (e.g. DICOM with no extension)
        if not files:
            continue
        path = os.path.join(folder, sorted(files)[0])
        try:
            if path.lower().endswith((".dcm", ".dicom")):
                arr = np.squeeze(pydicom.dcmread(path).pixel_array)
            else:
                # Try pydicom first for extensionless files (often DICOM), then image
                try:
                    arr = np.squeeze(pydicom.dcmread(path).pixel_array)
                except Exception:
                    arr = np.squeeze(plt.imread(path))
                if arr.ndim > 2:
                    arr = arr[..., :3] if arr.shape[-1] >= 3 else arr[..., 0]
        except Exception:
            continue
        if target_shape is None:
            target_shape = arr.shape
        if arr.shape == target_shape:
            slices.append(arr)
    if not slices:
        return None
    return np.stack(slices, axis=0)


# Load raw data: single DICOM file or folder of numbered subfolders (list of files)
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
    # Volume from list of files (numbered subfolders); None indicates no single file
    dicom_path = None
    pixel_array = load_volume_from_numbered_folders(root_path, exclude=("labels",))
    if pixel_array is None:
        path_err = ""
        try:
            raw_list = os.listdir(root_path) if os.path.isdir(root_path) else []
        except OSError as e:
            raw_list = []
            path_err = f" (os error: {e})"
        else:
            pass
        subdirs = [d for d in raw_list if os.path.isdir(os.path.join(root_path, d))]
        hint = f" Subfolders seen: {subdirs[:10]}{'...' if len(subdirs) > 10 else ''}." if subdirs else f" Directory listing: {raw_list[:15]}."
        if subdirs:
            sample_dir = os.path.join(root_path, subdirs[0])
            try:
                hint += f" First folder contents: {os.listdir(sample_dir)}"
            except OSError:
                pass
        raise FileNotFoundError(
            f"No volume found under {root_path}.{path_err}"
            f" Check that path exists and subfolders contain image files (.dcm or extensionless).{hint}"
        )

# Load labels (NIfTI from ITK-SNAP); fallback for numbered-folder layout
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


def overlay_volume(alpha=0.5):
    """Interactive viewer: slider to move through slices."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.12)
    slice_idx = n_slices // 2
    rgb = pixel_array[slice_idx] / rgb_max
    lab = labels_aligned[slice_idx]
    im_rgb = ax.imshow(rgb)
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
