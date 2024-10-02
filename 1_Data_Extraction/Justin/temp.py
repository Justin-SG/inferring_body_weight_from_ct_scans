from pathlib import Path
import numpy as np
import pandas as pd
from totalsegmentator.python_api import totalsegmentator
import xml.etree.ElementTree as ET
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor

PATH_TO_DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"
PATH_TO_TEMP_DIR = PATH_TO_DATA_DIR / "temp"
PATH_TO_SEGMENTATION_DIR = PATH_TO_TEMP_DIR / "segmentation"
PATH_TO_SEGMENTATION_DF = PATH_TO_DATA_DIR / "segmentation_df.feather"

def segment_nifti_image():
    # load segmentation_df.feather
    segmentation_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)
    input_path = segmentation_df.iloc[0]["SliceDirectory"]
    nifti_image = totalsegmentator(input_path, output=None, fastest=True)

    # store nibabel image to nifti file
    nifti_image.to_filename(PATH_TO_SEGMENTATION_DIR / "test.nii.gz")
    return nifti_image


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



def count_non_zero_pixels(data, label):
    return np.sum(data == label)

def processNiftiImage(image):
    labels = getLabelsFromNiftiImage(image)
    data = image.get_fdata()
    
    # Count the number of non-zero pixels in the segmentation image for each label
    non_zero_pixels = {label: np.count_nonzero(data == labels[label]) for label in labels}
    
    return non_zero_pixels


def load_nifti_image():
    nif_image = segment_nifti_image()
    # path = PATH_TO_SEGMENTATION_DIR / "test.nii.gz"
    # nif_image = nib.load(path)
    
    processNiftiImage(nif_image)

if __name__ == "__main__":
    load_nifti_image()