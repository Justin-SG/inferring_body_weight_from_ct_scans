import sys
import pandas as pd
import numpy as np
from Transforms import Transforms
import torch
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path

# Adding Project Paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model' / 'Flo'

sys.path.append(str(project_dir / '3_Data_Preparation'))
from CT_Dataset import CtScanDataset


def create_directory_if_not_exist(path):
    """Creates directories if they do not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Create my model folder
    create_directory_if_not_exist(model_dir)

    query = 'BodyPart == "Stamm"'

    axial_dataset = CtScanDataset(df_query=query, transform=Transforms.axial_projection_resnet_transforms())
    coronal_dataset = CtScanDataset(df_query=query, transform=Transforms.coronal_projection_resnet_transforms())
    sagittal_dataset = CtScanDataset(df_query=query, transform=Transforms.sagittal_projection_resnet_transforms())

    import matplotlib.pyplot as plt

    plt.imshow(axial_dataset[0][0][0, :, :], cmap='bone')
    plt.show()
    plt.imshow(coronal_dataset[0][0][0, :, :], cmap='bone')
    plt.show()
    plt.imshow(sagittal_dataset[0][0][0, :, :], cmap='bone')
    plt.show()





