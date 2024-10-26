import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Configure the logger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PATH_TO_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"
PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"
PATH_TO_BINCOUNT_DF = PATH_TO_DATA_DIR / "bincount_df.feather"

# https://en.wikipedia.org/wiki/Hounsfield_scale
BINS = {
    "Air": (-1024, -976),  # ±24 HU for air
    "Fat": (-125, -85),  # ±5 HU for soft tissue like fat
    "Soft tissue on contrast CT": (95, 305),  # ±5 HU for soft tissue
    "Bone Cancellous": (290, 410),  # ±10 HU for cancellous bone
    "Bone Cortical": (490, 1910),  # ±10 HU for cortical bone
    "Lung Parenchyma": (-705, -595),
    "Kidney": (15, 50),
    "Liver": (49, 71),  # ±6 HU for liver
    "Lymph nodes": (5, 25),
    "Muscle": (30, 60),
    "Thymus (Children)": (15, 45),
    "Thymus (Adolescents)": (15, 125),
    "White matter": (15, 35),
    "Grey matter": (32, 50),
}


def readCleanDicomDataFrame():
    logger.info(f"Loading DICOM metadata from '{PATH_TO_CLEANED_DICOM_DF}'")
    dicom_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)

    return dicom_df


def getOrCreateBincountDataFrame():
    if os.path.exists(PATH_TO_BINCOUNT_DF):
        logger.info(f"Loading bincount DataFrame from '{PATH_TO_BINCOUNT_DF}'")
        bincount_df = pd.read_feather(PATH_TO_BINCOUNT_DF)
        return bincount_df

    bincount_df = pd.DataFrame()
    return bincount_df


def processBincount(bincount_df, scan_metadata):
    # Already processed this scan, return the existing bincount_df
    if ( len(bincount_df) != 0 and scan_metadata.StudyInstanceUID in bincount_df["StudyInstanceUID"].values ):
        return bincount_df

    pixel_array = np.load(scan_metadata.PixelArrayFile).flatten()
    pixel_array = (
        pixel_array * scan_metadata.RescaleSlope
    ) + scan_metadata.RescaleIntercept

    bin_counts = {key: 0 for key in BINS.keys()}

    for label, (lower_bound, upper_bound) in BINS.items():
        mask = (pixel_array >= lower_bound) & (pixel_array <= upper_bound)
        bin_counts[label] = np.sum(mask)

    temp_df = pd.DataFrame([bin_counts])
    bincount_df = pd.concat([bincount_df, temp_df], ignore_index=True)
    return bincount_df


def main():
    dicom_df = readCleanDicomDataFrame()
    bincount_df = getOrCreateBincountDataFrame()

    for i in tqdm(range(2), desc="Processing scans"):
    # for i in tqdm(range(len(dicom_df)), desc="Processing scans"):
        bincount_df = processBincount(bincount_df, dicom_df.iloc[i])

    bincount_df.to_feather(PATH_TO_BINCOUNT_DF, version=2, compression="zstd")
    logger.info(f"DataFrame saved to '{PATH_TO_BINCOUNT_DF}' successfully!")


if __name__ == "__main__":
    main()
