import os
import pandas as pd
from pathlib import Path
import sys

# Configure the logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PATH_TO_DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"
PATH_TO_BINCOUNT_DF = PATH_TO_DATA_DIR / "bincount_df.feather"

# https://en.wikipedia.org/wiki/Hounsfield_scale
BINS = {
    "Air": (-1000, -1000),
    "Fat": (-120, -90),
    "Soft tissue on contrast CT": (100, 300),
    "Bone Cancellous": (300, 400),
    "Bone Cortical": (500, 1900),
    "Subdural hematoma (First hours)": (75, 100),
    "Subdural hematoma (After 3 days)": (65, 85),
    "Subdural hematoma (After 10–14 days)": (35, 40),
    "Other blood Unclotted": (13, 50),
    "Other blood Clotted": (50, 75),
    "Pleural effusion Transudate": (2, 15),
    "Pleural effusion Exudate": (4, 33),
    "Other fluids Chyle": (-30, -30),
    "Water": (0, 0),
    "Urine": (-5, 15),
    "Bile": (-5, 15),
    "CSF": (15, 15),
    "Abscess/Pus": (0, 45),
    "Mucus": (0, 130),
    "Lung Parenchyma": (-700, -600),
    "Kidney": (20, 45),
    "Liver": (54, 66),  # 60 ± 6
    "Lymph nodes": (10, 20),
    "Muscle": (35, 55),
    "Thymus (Children)": (20, 40),
    "Thymus (Adolescents)": (20, 120),
    "White matter": (20, 30),
    "Grey matter": (37, 45),
    "Gallstone Cholesterol stone": (30, 100),
    "Gallstone Bilirubin stone": (90, 120),
    "Foreign body Windowpane glass": (500, 500),
    "Foreign body Aluminum, tarmac, etc.": (2100, 2300),
    "Foreign body Limestone": (2800, 2800),
    "Foreign body Copper": (14000, 14000),
    "Foreign body Silver": (17000, 17000),
    "Foreign body Steel": (20000, 20000),
    "Foreign body Gold/Steel/Brass": (30000, 30000)
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

def main():
    dicom_df = readCleanDicomDataFrame()
    bincount_df = getOrCreateBincountDataFrame()
    
    
    return

if __name__ == "__main__":
    main()