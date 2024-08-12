import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class CtScanDataset(Dataset):
    def __init__(self, dicom_df_path, df_query=None, transform=None):
        """
        :param dicom_df_path: Path to the dataframes containing the cleaned DICOM metadata
        :param df_query: query to apply to the DICOM metadata (eg. query only abdomen scans)
        :param transform: transformations to apply to the scan arrays
        """
        self.data_path = Path(dicom_df_path).resolve()
        # Load the DICOM metadata
        self.dicom_df = pd.read_feather(self.data_path.joinpath('cleaned_dicom_df.feather'))

        # Apply query if given
        if df_query:
            self.dicom_df = self.dicom_df.query(df_query)

        # Apply transformations
        self.transform = transform

    def __len__(self):
        return len(self.dicom_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan = self.dicom_df.iloc[idx]
        weight = scan['PatientWeight']

        # weight to float -> necessary for loss calculation
        weight = np.float32(weight)

        pixel_array = np.load(self.data_path.joinpath(f'PixelArray/{scan["PixelArrayFile"]}'))

        if self.transform:
            pixel_array = self.transform(pixel_array)

        return pixel_array, weight

