import pandas as pd
from pathlib import Path
import sys
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def data_cleaning(dicom_df):
    logger.info("Cleaning the DICOM metadata...")
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].fillna('-1')  # Fill missing values with -1
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].str.replace('Y', '').astype(int)  # Remove 'Y' and convert to int
    dicom_df.loc[:, "BodyPart"] = dicom_df["ProcedureCodeSequence.CodeMeaning"].str.split(".", expand=True)[2]  # Extract body part from ProcedureCodeSequence.CodeMeaning
    dicom_df.loc[:, "PixelSpacing"] = dicom_df["PixelSpacing"].apply(lambda x: x[0])  # Extract the first value of PixelSpacing (same everywhere)


if __name__ == '__main__':
    # Load the DICOM metadata frames and concatenate them
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / 'Data'
    dicom_df_path = data_dir / 'dicom_df.feather'
    logger.info(f"Loading DICOM metadata from '{dicom_df_path}'")
    dicom_df = pd.read_feather(data_dir / 'dicom_df.feather')

    # Cleaning
    data_cleaning(dicom_df)

    # Get only the columns we need
    logger.info("Selecting the necessary columns...")
    dicom_df = dicom_df[["PatientId", "PatientAge", "PatientSex", "PatientWeight", "PatientSize", "BodyPart", "Rows",
                         "Columns", "SliceCount", "PixelSpacing", "SliceThickness", "PixelArrayFile"]]

    # Save the cleaned dataframe
    cleaned_dicom_df_path = data_dir / 'cleaned_dicom_df.feather'
    logger.info(f"Saving cleaned DICOM metadata to '{cleaned_dicom_df_path}'")
    dicom_df.to_feather(cleaned_dicom_df_path, version=2, compression='zstd')
