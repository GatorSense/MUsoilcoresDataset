import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider

from utils import get_paths, load_volume, load_labels, load_label_descriptions

# Volumes: (1) single DICOM file (set file_name) or (2) list of DICOM files in root_path (file_name = None).
# root_path = r"..//MU//SOIL64_8_10_20"
# file_name = "SOIL_FRIT_8_10_20Series_003_Bone 0.5 SCAN 1.2"  # None = list of DICOM files in root (e.g. 00000, 00010); else single .dcm file name

root_path = r"..\MU\DICOM_01\0000\002"
file_name = None

base_path = "."
full_root, _dicom_path, labels_path, descriptions_path, _ = get_paths(root_path, file_name, base_path)
if not os.path.isfile(labels_path):
    labels_path = os.path.join(full_root, "labels", "corelabels.nii.gz")
    descriptions_path = os.path.join(full_root, "labels", "descriptions.txt")

pixel_array, volume_from_list_of_files = load_volume(root_path, file_name, base_path)
if pixel_array is None:
    path_err = ""
    try:
        raw_list = os.listdir(full_root) if os.path.isdir(full_root) else []
    except OSError as e:
        raw_list = []
        path_err = f" (os error: {e})"
    files_in_root = [x for x in raw_list if os.path.isfile(os.path.join(full_root, x))]
    hint = f" Files in root: {files_in_root[:10]}{'...' if len(files_in_root) > 10 else ''}." if files_in_root else f" Directory listing: {raw_list[:15]}."
    raise FileNotFoundError(
        f"No volume found under {full_root}.{path_err} "
        f"Expected DICOM files in root (e.g. 00000, 00010).{hint}"
    )

labels_dir = os.path.join(full_root, "labels")
labels_img, labels_array = load_labels(labels_path, fallback_labels_dir=labels_dir)
label_descriptions = load_label_descriptions(descriptions_path)

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
volume_4d = pixel_array.ndim == 4  # (Z, Y, X, C); display uses grayscale slice


def _slice_for_display(i):
    """Return 2D slice (Y, X) for display; convert 4D (Y, X, C) to grayscale if needed."""
    s = pixel_array[i]
    if s.ndim == 3:
        s = np.tensordot(s, [0.299, 0.587, 0.114], axes=(-1, 0))
    return s / rgb_max


def overlay_volume(alpha=0.5):
    """Interactive viewer: slider to move through slices."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.12)
    slice_idx = n_slices // 2
    rgb = _slice_for_display(slice_idx)
    lab = labels_aligned[slice_idx]
    scan_cmap = "gray" if volume_from_list_of_files or volume_4d else None
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
        im_rgb.set_data(_slice_for_display(i))
        im_lab.set_data(np.ma.masked_where(labels_aligned[i] == 0, labels_aligned[i]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    overlay_volume()
