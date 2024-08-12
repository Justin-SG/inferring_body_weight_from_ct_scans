import pandas as pd
import os
import argparse
from pathlib import Path


def data_cleaning(dicom_df):
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].fillna('-1')  # Fill missing values with -1
    dicom_df.loc[:, "PatientAge"] = dicom_df["PatientAge"].str.replace('Y', '').astype(int)  # Remove 'Y' and convert to int
    dicom_df.loc[:, "BodyPart"] = dicom_df["ProcedureCodeSequence.CodeMeaning"].str.split(".", expand=True)[2]  # Extract body part from ProcedureCodeSequence.CodeMeaning


if __name__ == '__main__':
    # Get Program Arguments for data path and output path
    parser = argparse.ArgumentParser(description='Clean the dataframe')
    parser.add_argument('data_path', type=str, help='Path to the dataframes containing the DICOM metadata')
    parser.add_argument('output_path', type=str, help='Path to save the cleaned dataframe')
    args = parser.parse_args()

    # Load the DICOM metadata frames and concatenate them
    dicom_df_path = Path(args.data_path).resolve()
    chunks = dicom_df_path.glob('dicom_df_*.feather')
    dicom_dfs = [pd.read_feather(f) for f in chunks]
    dicom_df = pd.concat(dicom_dfs)

    # Cleaning
    data_cleaning(dicom_df)

    # Get only the columns we need
    dicom_df = dicom_df[["PatientId", "PatientAge", "PatientSex", "PatientWeight", "PatientSize", "BodyPart", "Rows",
                         "Columns", "SliceCount", "PixelSpacing", "SliceThickness", "PixelArrayFile"]]

    output_path = Path(args.output_path).resolve()
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Save the cleaned dataframe
    dicom_df.to_feather(output_path.joinpath('cleaned_dicom_df.feather'), version=2, compression='zstd')
