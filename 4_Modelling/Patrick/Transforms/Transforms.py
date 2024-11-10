import numpy as np
from .CustomTransforms import *
from torchvision.transforms import v2
import torchio as tio

HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900


def cnn_3d():
    return tio.Compose([
    # Convert the input to a tensor
    v2.Lambda(lambda x: x[:, :, np.newaxis]),
    # dimesion D x W x C x H -> C x W x H x D
    tio.Lambda(lambda x: x.permute(2, 1, 3, 0)),
    # Normalize the pixel values to the range [0, 1]
    tio.RescaleIntensity((HOUNSFIELD_AIR, HOUNSFIELD_BONE), (0, 1)),

    # Crop or Pad the input to the target size (600, 512, 512)
    tio.CropOrPad((512, 512, 600)),

    # Scale images to 224x224
    tio.Resize((224, 224, 224)),
    # Normalize with mean and std
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
