import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class CtScanDataset(Dataset):
    def __init__(self, dicom_df_path, transform=None):
        """
        :param dicom_df_path: Path to the dataframes containing the DICOM metadata
        :param transform: transformations to apply to the scan arrays
        """
        self.data_path = Path(dicom_df_path).resolve()
        # Load the DICOM metadata frames and concatenate them
        chunks = self.data_path.glob('dicom_df_*.feather')
        dicom_dfs = [pd.read_feather(f) for f in chunks]
        self.dicom_df = pd.concat(dicom_dfs)
        self.transform = transform

    def __len__(self):
        return len(self.dicom_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan = self.dicom_df.iloc[idx]
        weight = scan['PatientWeight']

        # weight to float
        weight = np.float32(weight)

        pixel_array = np.load(self.data_path.joinpath(f'PixelArray/{scan["PixelArrayFile"]}'))

        if self.transform:
            pixel_array = self.transform(pixel_array)

        return pixel_array, weight