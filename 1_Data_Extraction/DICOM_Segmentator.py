from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from pathlib import Path
import sys
from totalsegmentator.python_api import totalsegmentator
import logging
import nibabel as nib
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

PATH_TO_DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"
PATH_TO_SEGMENTATION_DIR = PATH_TO_DATA_DIR / "temp" / "segmentation"
PATH_TO_SEGMENTATION_DF = PATH_TO_DATA_DIR / "segmentation_df.feather"
NIFTI_FILE_ENDING = ".nii.gz"

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def readCleanDicomDataFrame():
    logger.info(f"Loading DICOM metadata from '{PATH_TO_CLEANED_DICOM_DF}'")
    dicom_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)

    return dicom_df


def getOrCreateSegmentationDataFrame():
    if os.path.exists(PATH_TO_SEGMENTATION_DF):
        logger.info(f"Loading segmentation DataFrame from '{PATH_TO_SEGMENTATION_DF}'")
        segmentation_df = pd.read_feather(PATH_TO_SEGMENTATION_DF)
        return segmentation_df

    segmentation_df = pd.DataFrame()
    return segmentation_df


def processNiftiFile(nifti_file_path):
    nifti_file = nib.load(nifti_file_path)

    segmentation_data = nifti_file.get_fdata()

    non_zero_pixels = np.count_nonzero(segmentation_data)

    # Create column name using filename without .nii.gz
    column_name = nifti_file_path.name[: -len(NIFTI_FILE_ENDING)]

    os.remove(nifti_file_path)
    return column_name, non_zero_pixels


def getLabelsFromNiftiImage(image):
    xml_content = image.header.extensions[0].get_content()
    root = ET.fromstring(xml_content)
    labels = root.findall(".//LabelTable/Label")

    labels_dict = {}
    for label in labels:
        value = int(label.get("Key"))
        key = label.text
        labels_dict[key] = value

    return labels_dict


def calculateSegmentationVolumes(image):
    labels = getLabelsFromNiftiImage(image)
    data = image.get_fdata()

    # Count the number of non-zero pixels in the segmentation image for each label
    non_zero_pixels = {
        label: np.count_nonzero(data == labels[label]) for label in labels
    }

    return non_zero_pixels


def processSegmentation(segmentation_df, scan):
    input_path = scan.SliceDirectory

    if (
        len(segmentation_df) != 0
        and input_path in segmentation_df["SliceDirectory"].values
    ):
        return segmentation_df

    nifti_image = totalsegmentator(input_path, output=None, fastest=True)
    segmentation_volumes = calculateSegmentationVolumes(nifti_image)
    segmentation_volumes["SliceDirectory"] = input_path

    temp_df = pd.DataFrame([segmentation_volumes])
    segmentation_df = pd.concat([segmentation_df, temp_df], ignore_index=True)
    return segmentation_df

def main():
    dicom_df = readCleanDicomDataFrame()
    segmentation_df = getOrCreateSegmentationDataFrame()
    
    # for i in tqdm(range(2), desc="Processing scans"):
    for i in tqdm(range(len(dicom_df)), desc="Processing scans"):
        segmentation_df = processSegmentation(segmentation_df, dicom_df.iloc[i])
        # Every 10 scans, save the segmentation_df to a feather file
        if i % 10 == 0:
            segmentation_df.to_feather(PATH_TO_SEGMENTATION_DF, version=2, compression="zstd")
            logger.info(f"Temporary DataFrame saved to '{PATH_TO_SEGMENTATION_DF}' successfully!")
        
    segmentation_df.to_feather(PATH_TO_SEGMENTATION_DF, version=2, compression="zstd")
    logger.info(f"DataFrame saved to '{PATH_TO_SEGMENTATION_DF}' successfully!")


if __name__ == "__main__":
    main()
