import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, ViT_B_16_Weights, ViT_B_32_Weights
from pathlib import Path

# Adding Project Paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model' / 'Flo'

sys.path.append(str(project_dir / '3_Data_Preparation'))

from CT_Datasets import CtScanDataset, CtScanDatasetExtended
from Transforms import Transforms
from CustomModels import CTWeightRegressor2D


def create_directory_if_not_exist(path):
    """Creates directories if they do not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_model(model_name, pretrained=True):
    """Load a pretrained model."""
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet152":
        model = models.resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("Model name not recognized.")
    return model


def train_and_validate(model, train_loader, val_loader, num_epochs=25, extended=False):
    """Train and validate the model over a number of epochs."""
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            if extended:
                inputs, additional_params, targets = data
                outputs = model(inputs, additional_params)
            else:
                inputs, targets = data
                outputs = model(inputs)

            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                if extended:
                    inputs, additional_params, targets = data
                    outputs = model(inputs, additional_params)
                else:
                    inputs, targets = data
                    outputs = model(inputs)

                loss = criterion(outputs, targets.unsqueeze(1))
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return train_losses, val_losses


def plot_loss_curves(train_losses, val_losses, model_name, dataset_name):
    """Plot the loss curves for training and validation."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curves for {model_name} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Create model folder
    create_directory_if_not_exist(model_dir)

    # Define the datasets
    query = 'BodyPart == "Stamm"'
    datasets = {
        "axial": CtScanDataset(df_query=query, transform=Transforms.axial_projection_imagenet_transforms()),
        "coronal": CtScanDataset(df_query=query, transform=Transforms.coronal_projection_imagenet_transforms()),
        "sagittal": CtScanDataset(df_query=query, transform=Transforms.sagittal_projection_imagenet_transforms())
    }

    datasets_extended = {
        "coronal_scaling": CtScanDatasetExtended(df_query=query,
                                                 pixel_array_transform=Transforms.coronal_projection_imagenet_transforms(),
                                                 imagenet_scaling_factor=True),
        "sagittal_scaling": CtScanDatasetExtended(df_query=query,
                                                  pixel_array_transform=Transforms.sagittal_projection_imagenet_transforms(),
                                                  imagenet_scaling_factor=True)
    }

    # Define the models
    models_dict = {
        "resnet18": get_model("resnet18"),
        "resnet50": get_model("resnet50"),
        "resnet152": get_model("resnet152"),
        "vit_b_16": get_model("vit_b_16"),
        "vit_b_32": get_model("vit_b_32")
    }

    # Train and validate each model on each dataset (original)
    for dataset_name, dataset in datasets.items():
        # Split dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset,
                                                  [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
                                                  generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for model_name, model in models_dict.items():
            print(f'Training {model_name} on {dataset_name} dataset...')
            train_losses, val_losses = train_and_validate(model, train_loader, val_loader, num_epochs=10)

            # Plot the loss curves
            plot_loss_curves(train_losses, val_losses, model_name, dataset_name)

            # Save the model
            model_save_path = model_dir / f'{model_name}_{dataset_name}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved {model_name} on {dataset_name} dataset to {model_save_path}')

    # Train and validate each custom model on each extended dataset
    for dataset_name, dataset in datasets_extended.items():
        # Split dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset,
                                                  [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
                                                  generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for model_name, backend_model in models_dict.items():
            print(f'Training {model_name} with custom model on {dataset_name} dataset...')

            # Initialize the custom model
            num_additional_params = 1  # In this case, only imagenet_scaling_factor is used
            model = CTWeightRegressor(backend_model, num_additional_params=num_additional_params,
                                      fc_layers=[512, 256, 128])

            # Train and validate the custom model
            train_losses, val_losses = train_and_validate(model, train_loader, val_loader, num_epochs=10, extended=True)

            # Plot the loss curves
            plot_loss_curves(train_losses, val_losses, model_name, dataset_name)

            # Save the model
            model_save_path = model_dir / f'{model_name}_scaling_{dataset_name}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved {model_name} with scaling on {dataset_name} dataset to {model_save_path}')