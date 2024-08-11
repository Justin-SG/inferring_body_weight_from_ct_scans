import sys
import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms


# include ../../3_Datapreparation
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../3_Data_Preparation')))
from CT_Dataset import CtScanDataset

# define transforms for resnet18
transforms = transforms.Compose([
    # change the data type to uint32
    transforms.Lambda(lambda x: x.astype(np.uint32)),
    # sum up all the pixel values over an axis and project them onto a 2D plane (Axial Projection)
    transforms.Lambda(lambda x: x.sum(axis=0)),
    # add color channel
    transforms.Lambda(lambda x: x[np.newaxis, :, :]),
    # repeat color channel 3 times
    transforms.Lambda(lambda x: np.repeat(x, 3, axis=0)),
    # Resize images to 224x224
    #transforms.Resize((224, 224)),
    # Convert Array to tensor and normalize to [0, 1]
    #transforms.ToTensor(),
    # Normalize with mean and std
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = CtScanDataset("inferring_body_weight_from_ct_scans/1_Data_Extraction/Data", transform=None)
print(len(dataset))
print(dataset[0])

print(dataset[0][0])
dataset[0][0].astype(np.uint32)
print(dataset[0][0])
