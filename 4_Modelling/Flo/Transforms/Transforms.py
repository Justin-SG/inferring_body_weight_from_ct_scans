import numpy as np
from .CustomTransforms import *
from torchvision.transforms import v2


def axial_projection_resnet_transforms():
    return v2.Compose([
        # change the data type to uint32 -> prevents overflow
        v2.Lambda(lambda x: x.astype(np.float32)),
        # sum up all the pixel values over an axis and project them onto a 2D plane (Axial Projection)
        v2.Lambda(lambda x: x.sum(axis=0)),
        # add color channel dimension
        v2.Lambda(lambda x: x[:, :, np.newaxis]),
        # repeat color channel 3 times (RGB)
        v2.Lambda(lambda x: np.repeat(x, 3, axis=2)),
        # To PIL Image (required for torchvision transforms)
        v2.ToImage(),
        # Scale images to 224x224
        v2.Resize((224, 224), antialias=True),
        # Normalize tensor values to range [0, 1] with lambda function
        ScaleTo01(),
        # Normalize with mean and std
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def coronal_projection_resnet_transforms():
    return v2.Compose([
        # change the data type to uint32 -> prevents overflow
        v2.Lambda(lambda x: x.astype(np.float32)),
        # sum up all the pixel values over an axis and project them onto a 2D plane (Coronal Projection)
        v2.Lambda(lambda x: x.sum(axis=1)),
        # add color channel dimension
        v2.Lambda(lambda x: x[:, :, np.newaxis]),
        # repeat color channel 3 times (RGB)
        v2.Lambda(lambda x: np.repeat(x, 3, axis=2)),
        # To PIL Image (required for torchvision transforms)
        v2.ToImage(),
        # Scale images to 224x224
        v2.Resize((224, 224), antialias=True),
        # Normalize tensor values to range [0, 1] with lambda function
        ScaleTo01(),
        # Normalize with mean and std
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def sagittal_projection_resnet_transforms():
    return v2.Compose([
        # change the data type to uint32 -> prevents overflow
        v2.Lambda(lambda x: x.astype(np.float32)),
        # sum up all the pixel values over an axis and project them onto a 2D plane (Sagittal Projection)
        v2.Lambda(lambda x: x.sum(axis=2)),
        # add color channel dimension
        v2.Lambda(lambda x: x[:, :, np.newaxis]),
        # repeat color channel 3 times (RGB)
        v2.Lambda(lambda x: np.repeat(x, 3, axis=2)),
        # To PIL Image (required for torchvision transforms)
        v2.ToImage(),
        # Scale images to 224x224
        v2.Resize((224, 224), antialias=True),
        # Normalize tensor values to range [0, 1] with lambda function
        ScaleTo01(),
        # Normalize with mean and std
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])