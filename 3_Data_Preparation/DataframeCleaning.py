import pandas as pd
from pathlib import Path


def data_cleaning(dicom_df):
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].fillna('-1')  # Fill missing values with -1
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].str.replace('Y', '').astype(int)  # Remove 'Y' and convert to int
    dicom_df.loc[:, "BodyPart"] = dicom_df["ProcedureCodeSequence.CodeMeaning"].str.split(".", expand=True)[2]  # Extract body part from ProcedureCodeSequence.CodeMeaning


if __name__ == '__main__':
    # Load the DICOM metadata frames and concatenate them
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / 'Data'
    dicom_df = pd.read_feather(data_dir / 'dicom_df.feather')

    # Cleaning
    data_cleaning(dicom_df)

    # Get only the columns we need
    dicom_df = dicom_df[["PatientId", "PatientAge", "PatientSex", "PatientWeight", "PatientSize", "BodyPart", "Rows",
                         "Columns", "SliceCount", "PixelSpacing", "SliceThickness", "PixelArrayFile"]]

    # Save the cleaned dataframe
    dicom_df.to_feather(data_dir / 'cleaned_dicom_df.feather', version=2, compression='zstd')
