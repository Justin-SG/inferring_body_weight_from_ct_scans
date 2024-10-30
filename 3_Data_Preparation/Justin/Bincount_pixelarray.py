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
PATH_TO_BINCOUNT_HU_DF = PATH_TO_DATA_DIR / "bincount_HU_df.feather"
PATH_TO_BINCOUNT_STEP_75_DF = PATH_TO_DATA_DIR / "bincount_STEP_75_df.feather"
PATH_TO_BINCOUNT_STEP_150_DF = PATH_TO_DATA_DIR / "bincount_STEP_150_df.feather"

# https://en.wikipedia.org/wiki/Hounsfield_scale
BINS_HU = {
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

# Bins from -1000 to 500 with a step of 75/150. Dont apply RescaleSlope and RescaleIntercept, so it starts from 0 instead of -1024
BINS_STEP_75 = {str(i): (i, i + 100) for i in range(0, 1500, 75)}

BINS_STEP_150 = {str(i): (i, i + 100) for i in range(0, 1500, 150)}


def readCleanDicomDataFrame():
    logger.info(f"Loading DICOM metadata from '{PATH_TO_CLEANED_DICOM_DF}'")
    dicom_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)

    return dicom_df


def getOrCreateBincountDataFrame(path_to_df):
    if os.path.exists(path_to_df):
        logger.info(f"Loading bincount DataFrame from '{path_to_df}'")
        bincount_df = pd.read_feather(path_to_df)
        return bincount_df

    bincount_df = pd.DataFrame()
    return bincount_df


def processBincount(bincount_df, scan_metadata, bins, apply_rescale_slope_intercept):
    # Already processed this scan, return the existing bincount_df
    if ( len(bincount_df) != 0 and scan_metadata.SeriesInstanceUID in bincount_df["SeriesInstanceUID"].values ):
        return bincount_df

    pixel_array = np.load(scan_metadata.PixelArrayFile).flatten()
    
    if apply_rescale_slope_intercept:
        pixel_array = (
            pixel_array * scan_metadata.RescaleSlope
        ) + scan_metadata.RescaleIntercept

    bin_counts = {key: 0 for key in bins.keys()}

    for label, (lower_bound, upper_bound) in bins.items():
        mask = (pixel_array >= lower_bound) & (pixel_array <= upper_bound)
        bin_counts[label] = np.sum(mask)

    temp_df = pd.DataFrame([bin_counts])
    temp_df["SeriesInstanceUID"] = scan_metadata.SeriesInstanceUID
    bincount_df = pd.concat([bincount_df, temp_df], ignore_index=True)
    return bincount_df

def main():
    dicom_df = readCleanDicomDataFrame()
    bincount_HU_df = getOrCreateBincountDataFrame(PATH_TO_BINCOUNT_HU_DF)
    bincount_STEP_75_df = getOrCreateBincountDataFrame(PATH_TO_BINCOUNT_STEP_75_DF)
    bincount_STEP_150_df = getOrCreateBincountDataFrame(PATH_TO_BINCOUNT_STEP_150_DF)

    for i in tqdm(range(2), desc="Processing scans"):
    # for i in tqdm(range(len(dicom_df)), desc="Processing scans"):
        bincount_HU_df = processBincount(bincount_HU_df, dicom_df.iloc[i], BINS_HU, True)
        bincount_STEP_75_df = processBincount(bincount_STEP_75_df, dicom_df.iloc[i], BINS_STEP_75, False)
        bincount_STEP_150_df = processBincount(bincount_STEP_150_df, dicom_df.iloc[i], BINS_STEP_150, False)

    bincount_HU_df.to_feather(PATH_TO_BINCOUNT_HU_DF, version=2, compression="zstd")
    logger.info(f"DataFrame saved to '{PATH_TO_BINCOUNT_HU_DF}' successfully!")
    
    bincount_STEP_75_df.to_feather(PATH_TO_BINCOUNT_STEP_75_DF, version=2, compression="zstd")
    logger.info(f"DataFrame saved to '{PATH_TO_BINCOUNT_STEP_75_DF}' successfully!")
    
    bincount_STEP_150_df.to_feather(PATH_TO_BINCOUNT_STEP_150_DF, version=2, compression="zstd")
    logger.info(f"DataFrame saved to '{PATH_TO_BINCOUNT_STEP_150_DF}' successfully!")


if __name__ == "__main__":
    main()
