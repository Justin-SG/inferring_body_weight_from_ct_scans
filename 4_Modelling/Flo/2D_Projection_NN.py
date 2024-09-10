import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, ViT_B_16_Weights, ViT_B_32_Weights
from pathlib import Path
import argparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adding Project Paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model' / 'Flo'
eval_dir = project_dir / '5_Evaluation' / 'Flo'
loss_curves_dir = project_dir / '5_Evaluation' / 'Flo' / 'Loss_Curves'

sys.path.append(str(project_dir / '3_Data_Preparation'))

from CT_Datasets import CtScanDataset, CtScanDatasetExtended
from Transforms import Transforms
from CustomModels import CtWeightRegressorAdditionalParams2D, CtMultipliedScaleWeightRegressor2D


def create_directory_if_not_exist(path):
    """Creates directories if they do not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_model(model_name, pretrained=True, with_regression_layer=False):
    """Load a pretrained model."""
    model_mapping = {
        "resnet18": models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None),
        "resnet50": models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None),
        "resnet152": models.resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None),
        "vit_b_16": models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None),
        "vit_b_32": models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT if pretrained else None)
    }

    if model_name not in model_mapping:
        raise ValueError(f"Model name '{model_name}' not recognized.")

    model = model_mapping[model_name]

    if with_regression_layer:
        if 'vit' in model_name:
            model.heads.head = nn.Linear(model.heads.head.in_features, 1)
        else:
            model.fc = nn.Linear(model.fc.in_features, 1)

    return model


def train_and_validate(model,
                       train_loader,
                       val_loader,
                       device,
                       num_epochs=25,
                       patience=5,
                       learning_rate=0.001,
                       with_additional_params=False,
                       multiplied_scaling_factor=False):
    """Train and validate the model over a number of epochs with early stopping."""
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    trained_epochs = 0

    # Move model to the device (GPU or CPU)
    model = model.to(device)

    # Record the start time of training
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            if with_additional_params or multiplied_scaling_factor:
                inputs, additional_params, targets = data
                inputs, additional_params, targets = inputs.to(device), additional_params.to(device), targets.to(device)
                outputs = model(inputs, additional_params)
            else:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        max_absolute_error = 0.0

        with torch.no_grad():
            for data in val_loader:
                if with_additional_params or multiplied_scaling_factor:
                    inputs, additional_params, targets = data
                    inputs, additional_params, targets = inputs.to(device), additional_params.to(device), targets.to(device)
                    outputs = model(inputs, additional_params)
                else:
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, targets.unsqueeze(1))
                running_val_loss += loss.item() * inputs.size(0)

                # Calculate max absolute error
                max_absolute_error = max(max_absolute_error, torch.max(torch.abs(outputs - targets.unsqueeze(1))).item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Max Abs Error: {max_absolute_error:.4f}')

        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                trained_epochs = epoch + 1  # Capture the number of epochs trained
                break
        trained_epochs = epoch + 1

    # Record the total training time
    training_time = time.time() - start_time

    return train_losses, val_losses, trained_epochs, max_absolute_error, training_time


def plot_and_save_loss_curves(train_losses, val_losses, model_name, dataset_name, loss_curves_dir):
    """Plot and save the loss curves for training and validation."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curves for {model_name} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to the evaluation directory
    plot_path = loss_curves_dir / f'{model_name}_{dataset_name}_loss_curve.png'
    plt.savefig(plot_path)
    logger.info(f"Saved loss curves to {plot_path}")


def model_exists(model_name, dataset_name, additional_params=None, multiplied_scaling_factor=False):
    """Check if the model already exists in the model directory."""
    model_suffix = "_scale_multiplied" if multiplied_scaling_factor else ""
    if additional_params:
        for param in additional_params:
            model_suffix += f"_{param}"

    model_path = model_dir / f'{model_name}{model_suffix}_{dataset_name}.pth'
    return model_path.exists(), model_path


def save_model_stats(stats, eval_dir):
    """Save model statistics to a CSV file."""
    stats_df = pd.DataFrame(stats)
    stats_file = eval_dir / 'model_statistics.csv'

    if stats_file.exists():
        existing_stats = pd.read_csv(stats_file)
        stats_df = pd.concat([existing_stats, stats_df], ignore_index=True)

    stats_df.to_csv(stats_file, index=False)


def train_model_on_dataset(model_name,
                           dataset_name,
                           dataset,
                           query,
                           num_epochs,
                           patience,
                           device,
                           batch_size=32,
                           learning_rate=0.001,
                           with_scaling_factor=False,
                           multiplied_scaling_factor=False,
                           additional_params=None):
    """Train a model on a specific dataset, with optional additional parameters."""
    train_dataset, val_dataset = random_split(
        dataset,
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_already_exists, model_path = model_exists(model_name,
                                                    dataset_name,
                                                    multiplied_scaling_factor=multiplied_scaling_factor,
                                                    additional_params=additional_params)

    if model_already_exists and not args.overwrite:
        logger.info(f'Skipping {model_name} on {dataset_name} dataset (model already exists).')
        return None

    if with_scaling_factor and multiplied_scaling_factor:
        model = CtMultipliedScaleWeightRegressor2D(
            get_model(model_name, pretrained=True),
            fc_layers=[128, 64, 32]
        )
    elif additional_params:
        num_additional_params = len(additional_params)
        model = CtWeightRegressorAdditionalParams2D(
            get_model(model_name, pretrained=True),
            num_additional_params=num_additional_params,
            fc_layers=[128, 64, 32]
        )
    else:
        model = get_model(model_name, pretrained=True, with_regression_layer=True)

    if with_scaling_factor:
        logger.info(f'Training {model_name} on {dataset_name} dataset (Scale multiplied: {multiplied_scaling_factor})...')
    else:
        logger.info(f'Training {model_name} on {dataset_name} dataset...')

    train_losses, val_losses, trained_epochs, max_absolute_error, training_time = train_and_validate(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        patience=patience,
        learning_rate=learning_rate,
        with_additional_params=bool(additional_params),
        multiplied_scaling_factor=multiplied_scaling_factor
    )

    # Save the model
    torch.save(model.state_dict(), model_path)
    logger.info(f'Saved {model_name} on {dataset_name} dataset to {model_path}')

    # Plot and save loss curves
    plot_and_save_loss_curves(train_losses, val_losses, model_name, dataset_name, loss_curves_dir)

    # Update stats with max_absolute_error and training_time
    stats = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "query": query,
        "with_scaling_factor": with_scaling_factor,
        "multiplied_scaling_factor": multiplied_scaling_factor,
        "additional_params": additional_params,
        "train_loss mean_absolute_error": train_losses[-1],
        "val_loss mean_absolute_error": val_losses[-1],
        "max_absolute_error": max_absolute_error,
        "epochs_trained": trained_epochs,
        "training_time": training_time
    }

    return stats


def main(args):
    # Create necessary directories
    create_directory_if_not_exist(model_dir)
    create_directory_if_not_exist(eval_dir)
    create_directory_if_not_exist(loss_curves_dir)

    # Use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define the datasets
    query = 'BodyPart == "Stamm"'
    datasets = {
        "axial": CtScanDataset(df_query=query, transform=Transforms.axial_projection_imagenet_transforms()),
        "coronal": CtScanDataset(df_query=query, transform=Transforms.coronal_projection_imagenet_transforms()),
        "sagittal": CtScanDataset(df_query=query, transform=Transforms.sagittal_projection_imagenet_transforms())
    }

    datasets_scaling_factor = {
        "coronal_scaling": CtScanDatasetExtended(df_query=query,
                                                 pixel_array_transform=Transforms.coronal_projection_imagenet_transforms(),
                                                 imagenet_scaling_factor=True),
        "sagittal_scaling": CtScanDatasetExtended(df_query=query,
                                                  pixel_array_transform=Transforms.sagittal_projection_imagenet_transforms(),
                                                  imagenet_scaling_factor=True)
    }

    datasets_scaling_thickness_spacing = {
        "axial_spacing_thickness_scaling": CtScanDatasetExtended(df_query=query,
                                                                 pixel_array_transform=Transforms.axial_projection_imagenet_transforms(),
                                                                 additional_features=['PixelSpacing', 'SliceThickness'],
                                                                 imagenet_scaling_factor=True),
        "coronal_spacing_thickness_scaling": CtScanDatasetExtended(df_query=query,
                                                                   pixel_array_transform=Transforms.coronal_projection_imagenet_transforms(),
                                                                   additional_features=['PixelSpacing',
                                                                                        'SliceThickness'],
                                                                   imagenet_scaling_factor=True),
        "sagittal_spacing_thickness_scaling": CtScanDatasetExtended(df_query=query,
                                                                    pixel_array_transform=Transforms.sagittal_projection_imagenet_transforms(),
                                                                    additional_features=['PixelSpacing',
                                                                                         'SliceThickness'],
                                                                    imagenet_scaling_factor=True)
    }

    backends = [
        "resnet18",
        "resnet50",
        #"resnet152",
        "vit_b_16",
        "vit_b_32"
    ]

    model_statistics = []

    # Train and validate each model on each dataset (no additional features)
    for dataset_name, dataset in datasets.items():
        for model_name in backends:
            stats = train_model_on_dataset(
                model_name, dataset_name, dataset,
                query=query,
                num_epochs=args.epochs, patience=args.patience,
                device=device
            )
            if stats:
                model_statistics.append(stats)
                save_model_stats(model_statistics, eval_dir)

    # Train and validate each scaled model on each dataset (with scaling factor)
    for dataset_name, dataset in datasets_scaling_factor.items():
        additional_params = ['scaling_factor']  # Example additional parameter

        for model_name in backends:
            stats = train_model_on_dataset(
                model_name, dataset_name, dataset,
                query=query,
                num_epochs=args.epochs, patience=args.patience,
                with_scaling_factor=True, additional_params=additional_params,
                device=device
            )
            if stats:
                model_statistics.append(stats)
                save_model_stats(model_statistics, eval_dir)

    # Train and validate each model with multiplied scaling factor on each dataset
    for dataset_name, dataset in datasets_scaling_factor.items():
        additional_params = ['scaling_factor']  # Example additional parameter

        for model_name in backends:
            stats = train_model_on_dataset(
                model_name, dataset_name, dataset,
                query=query,
                num_epochs=args.epochs, patience=args.patience,
                with_scaling_factor=True,
                multiplied_scaling_factor=True,
                additional_params=additional_params,
                device=device
            )
            if stats:
                model_statistics.append(stats)
                save_model_stats(model_statistics, eval_dir)

    # Train and validate each model with additional features on each dataset
    for dataset_name, dataset in datasets_scaling_thickness_spacing.items():
        additional_params = ['scaling_factor', 'pixel_spacing', 'slice_thickness']

        for model_name in backends:
            stats = train_model_on_dataset(
                model_name, dataset_name, dataset,
                query=query,
                num_epochs=args.epochs, patience=args.patience,
                with_scaling_factor=True, additional_params=additional_params,
                device=device
            )
            if stats:
                model_statistics.append(stats)
                save_model_stats(model_statistics, eval_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and validate CT scan models.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing models.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    main(args)
