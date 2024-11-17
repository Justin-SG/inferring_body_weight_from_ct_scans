import numpy as np
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import torchio as tio
import torch

HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900


def rescale_if_nonconstant(image, intensity_range=(HOUNSFIELD_AIR, HOUNSFIELD_BONE), target_range=(0, 1)):
    # Clip values to a defined range to remove outliers
    image = image.clip(intensity_range[0], intensity_range[1])

    # Check if the image has constant intensity
    if image.min() == image.max():
        # Normalize image.min to the target range
        normalized_value = ((image.min() - intensity_range[0]) / (intensity_range[1] - intensity_range[0])) * (target_range[1] - target_range[0]) + target_range[0]
        return torch.full_like(image, normalized_value)
    else:
        # Rescale intensity if there is variability in the image
        return tio.RescaleIntensity(intensity_range, target_range)(image)

def cnn_3d():
    return v2.Compose([
    # change the data type to uint32 -> prevents overflow
    v2.Lambda(lambda x: x.astype(np.float32)),
    # Convert the input to a tensor
    v2.Lambda(lambda x: x[:, :, np.newaxis]),
    # dimesion D x W x C x H -> C x W x H x D
    tio.Lambda(lambda x: x.permute(2, 1, 3, 0)),
    # Normalize the pixel values to the range [0, 1]
    # Conditionally rescale intensity if the values are non-constant
    #tio.Lambda(rescale_if_nonconstant),

    # Crop or Pad the input to the target size (600, 512, 512)
    tio.CropOrPad((512, 512, 600)),

    # Scale images to 224x224
    tio.Resize((224, 224, 225)),
    # Normalize with mean and std
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CNN3DPreprocessor:
    def __call__(self, x):
        # Change the data type to float32 to prevent overflow
        x = x.astype(np.float32)
        
        # Add a new axis to make the input compatible with expected format
        x = x[:, :, np.newaxis]

        # Convert to PyTorch tensor
        x = torch.from_numpy(x)
        
        # Permute dimensions from D x W x C x H to C x W x H x D
        x = np.transpose(x, (2, 1, 3, 0))  # Adjust to PyTorch tensor permutation if needed
        
        # Crop or pad the input to the target size (512, 512, 600)
        x = tio.CropOrPad((512, 512, 600))(x)
        
        # Resize images to (112, 112, 113)
        x = tio.Resize((112, 112, 113))(x)

        # Conditionally rescale intensity if the values are non-constant
        tio.Lambda(rescale_if_nonconstant),

        return x
    
class CNN3DPreprocessor2:
    def __call__(self, x):
        # Change the data type to float32 to prevent overflow
        x = x.astype(np.float32)
        
        # Add a new axis to make the input compatible with expected format
        x = x[:, :, np.newaxis]

        # Convert to PyTorch tensor
        x = torch.from_numpy(x)

        # Permute dimensions from D x W x C x H to C x W x H x D
        x = np.transpose(x, (2, 1, 3, 0))  # Adjust to PyTorch tensor permutation if needed
        
        # Resize images to (112, 112, 112)
        x = tio.Resize((112, 112, 112))(x)

        # Conditionally rescale intensity if the values are non-constant
        tio.Lambda(rescale_if_nonconstant),

        return x
