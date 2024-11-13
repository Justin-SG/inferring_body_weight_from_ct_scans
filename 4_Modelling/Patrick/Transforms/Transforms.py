import numpy as np
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import torchio as tio
import torch

HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900


def rescale_if_nonconstant(image, intensity_range=(HOUNSFIELD_AIR, HOUNSFIELD_BONE), target_range=(0, 1)):
    if image.min() == image.max():
        return image  # No rescaling needed if all values are the same
    else:
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
    def __init__(self):
        # Initialize normalization parameters only if needed, but no Compose is used.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, x):
        # Change the data type to float32 to prevent overflow
        x = x.astype(np.float32)
        
        # Add a new axis to make the input compatible with expected format
        x = x[:, :, np.newaxis]

        # Convert to PyTorch tensor
        #x = torch.from_numpy(x)
        
        # Permute dimensions from D x W x C x H to C x W x H x D
        x = np.transpose(x, (2, 1, 3, 0))  # Adjust to PyTorch tensor permutation if needed
        
        # Crop or pad the input to the target size (512, 512, 600)
        x = tio.CropOrPad((512, 512, 600))(x)
        
        # Resize images to (224, 224, 225)
        x = tio.Resize((224, 224, 225))(x)

        # Expand the single channel to 3 channels by repeating
        #x = x.repeat(3, 1, 1, 1)
        
        # Normalize using mean and std
        #x = F.normalize(x, mean=self.mean, std=self.std)
        
        return x
