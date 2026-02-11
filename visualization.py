import os
import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider


root_path = r"D:\jose_data\MU\SOIL64_8_10_20"
file_name = "SOIL_FRIT_8_10_20Series_003_Bone 0.5 SCAN 1.2"
dicom_path = os.path.join(root_path, file_name + ".dcm")
labels_path = os.path.join(root_path, "labels", file_name + "_corelabels" + ".nii.gz")
descriptions_path = os.path.join(root_path, "labels", file_name + "_descriptions" + ".txt")

# Load raw data (DICOM)
ds = pydicom.dcmread(dicom_path)
pixel_array = np.squeeze(ds.pixel_array)

# Load labels (NIfTI from ITK-SNAP)
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
