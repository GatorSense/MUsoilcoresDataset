"""
Open the DICOM file 00010 from the DICOM directory.
Path: MU/DICOM_03/0000/002/00010
Uses pydicom to read and parse the file as a DICOM dataset.
"""
import os
import pydicom
import matplotlib.pyplot as plt

# Base path to the DICOM series (adjust if your MU folder is elsewhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, r"..\MU\DICOM_03\0000\002"))
TARGET = "00010"
TARGET_PATH = os.path.join(BASE_PATH, TARGET)


def open_00010():
    """Load 00010 as a pydicom Dataset. Returns None if path missing or invalid."""
    if not os.path.exists(BASE_PATH):
        print(f"Base path not found: {BASE_PATH}")
        return None
    if not os.path.exists(TARGET_PATH):
        print(f"Target not found: {TARGET_PATH}")
        return None

    if os.path.isfile(TARGET_PATH):
        ds = pydicom.dcmread(TARGET_PATH)
        print(f"Read DICOM file: {TARGET_PATH}")
        return ds
    elif os.path.isdir(TARGET_PATH):
        # If 00010 is a folder, read first DICOM file inside
        entries = sorted(os.listdir(TARGET_PATH))
        for name in entries:
            path = os.path.join(TARGET_PATH, name)
            if os.path.isfile(path):
                try:
                    ds = pydicom.dcmread(path)
                    print(f"Read DICOM from directory: {path}")
                    return ds
                except Exception:
                    continue
        print(f"No DICOM file found in directory: {TARGET_PATH}")
        return None
    return None


if __name__ == "__main__":
    ds = open_00010()
    if ds is not None:
        print("Dataset:", ds)
        if hasattr(ds, "PatientName"):
            print("  PatientName:", ds.PatientName)
        if hasattr(ds, "SeriesDescription"):
            print("  SeriesDescription:", ds.SeriesDescription)
        if hasattr(ds, "pixel_array"):
            print("  pixel_array shape:", ds.pixel_array.shape)
            plt.imshow(ds.pixel_array, cmap="gray")
            plt.colorbar()
            plt.title("DICOM 00010")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("  No pixel data in this dataset.")
