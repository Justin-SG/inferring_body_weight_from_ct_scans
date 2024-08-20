import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class CtScanDataset(Dataset):
    def __init__(self, df_query=None, transform=None):
        """
        :param df_query: query to apply to the DICOM metadata (eg. query only abdomen scans)
        :param transform: transformations to apply to the scan arrays
        """
        project_dir = Path(__file__).resolve().parent.parent

        self.data_path = project_dir / 'Data'
        # Load the DICOM metadata
        self.dicom_df = pd.read_feather(self.data_path / 'cleaned_dicom_df.feather')

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


class CtScanDatasetExtended(Dataset):
    def __init__(self, df_query=None, additional_features=None, imagenet_scaling_factor=False, pixel_array_transform=None, additional_features_transform=None):
        """
        :param df_query: query to apply to the DICOM metadata (eg. query only abdomen scans)
        :param additional_features: additional features (df columns -> numeric only and lists will be multiple features) to concatenate with the scan arrays
        :param imagenet_scaling_factor: if True, the scaling factor for ImageNet pre-trained models will be added as an additional feature
        :param pixel_array_transform: transformations to apply to the scan arrays
        :param additional_features_transform: transformations to apply to the additional features
        """
        project_dir = Path(__file__).resolve().parent.parent

        self.data_path = project_dir / 'Data'
        # Load the DICOM metadata
        self.dicom_df = pd.read_feather(self.data_path / 'cleaned_dicom_df.feather')

        # Apply query if given
        if df_query:
            self.dicom_df = self.dicom_df.query(df_query)

        # Additional feature flags
        self.additional_features = additional_features
        self.imagenet_scaling_factor = imagenet_scaling_factor

        # Apply transformations
        self.transform = pixel_array_transform

        if additional_features_transform:
            self.additional_features_transforms = additional_features_transform

    def __len__(self):
        """Get the number of scans in the dataset."""
        return len(self.dicom_df)

    def __getitem__(self, idx):
        """Get the scan pixel array, additional features, and weight."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan = self.dicom_df.iloc[idx]

        # Get the weight
        weight = scan['PatientWeight']

        # weight to float -> necessary for loss calculation
        weight = np.float32(weight)

        # Load the pixel array
        pixel_array = np.load(self.data_path.joinpath(f'PixelArray/{scan["PixelArrayFile"]}'))

        if self.transform:
            pixel_array = self.transform(pixel_array)

        # Additional features
        additional_inputs = []
        if self.additional_features:
            for feature in self.additional_features:
                # If the feature is a list or array, add each element as a separate feature
                if isinstance(scan[feature], (list, np.ndarray)):
                    additional_inputs.extend(scan[feature])
                else:
                    additional_inputs.append(scan[feature])


        # Add ImageNet scaling factor if required
        if self.imagenet_scaling_factor:
            # calculate depth scaling factor
            slice_count = scan['SliceCount']
            scaling_factor = slice_count / 224 # ImageNet input size
            additional_inputs.append(scaling_factor)

        return pixel_array, additional_inputs, weight