import sys
import pandas as pd
import torch
from pathlib import Path
import logging
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import feather

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adding Project Paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model' / 'Flo'
eval_dir = project_dir / '5_Evaluation' / 'Flo'

sys.path.append(str(project_dir / '3_Data_Preparation'))

from CT_Datasets import CtScanDataset, CtScanDatasetExtended
from Transforms import Transforms
from CustomModels import CtWeightRegressorAdditionalParams2D, CtMultipliedScaleWeightRegressor2D

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
                                                               additional_features=['PixelSpacing', 'SliceThickness'],
                                                               imagenet_scaling_factor=True),
    "sagittal_spacing_thickness_scaling": CtScanDatasetExtended(df_query=query,
                                                                pixel_array_transform=Transforms.sagittal_projection_imagenet_transforms(),
                                                                additional_features=['PixelSpacing', 'SliceThickness'],
                                                                imagenet_scaling_factor=True)
}


# Function to parse model name and select the appropriate dataset
def parse_model_name(model_name):
    parts = model_name.replace(".pth", "").split("_")
    backend = parts[0]
    dataset_name = parts[-1]
    if  "spacing" in parts:
        dataset_name = "_".join(parts[-3:])
    elif "scaling" in parts:
        dataset_name = "_".join(parts[-2:])


    # Check for scale multiplied
    scale_multiplied = "scale_multiplied" in parts

    # Extract additional params
    additional_params = [param for param in parts if param not in ["scale_multiplied", backend, dataset_name]]

    # Select the dataset
    if dataset_name in datasets:
        dataset = datasets[dataset_name]
    elif dataset_name in datasets_scaling_factor:
        dataset = datasets_scaling_factor[dataset_name]
    elif dataset_name in datasets_scaling_thickness_spacing:
        dataset = datasets_scaling_thickness_spacing[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return backend, dataset, scale_multiplied, additional_params


# Function to load model
def load_model(model_path, backend, scale_multiplied, additional_params):
    if scale_multiplied:
        model = CtMultipliedScaleWeightRegressor2D(backend=backend)
    elif additional_params:
        model = CtWeightRegressorAdditionalParams2D(backend=backend, additional_params=additional_params)
    else:
        model = CtWeightRegressorAdditionalParams2D(backend=backend)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Prediction loop
def predict_and_save_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # Get all model files
    model_files = list(model_dir.glob("*.pth"))

    # Load any existing results to continue from where it crashed
    output_file = eval_dir / 'model_predictions.feather'
    if output_file.exists():
        logger.info("Loading existing results...")
        df_results = pd.read_feather(output_file)
        processed_models = df_results['model_name'].unique()
    else:
        df_results = pd.DataFrame(
            columns=['model_name', 'true_weight', 'predicted_weight', 'set_type', 'additional_params'])
        processed_models = []

    # Process each model
    for model_file in tqdm(model_files, desc="Processing models"):
        model_name = model_file.stem

        if model_name in processed_models:
            logger.info(f"Skipping already processed model {model_name}")
            continue

        # Parse the model name to get the right dataset
        backend, dataset, scale_multiplied, additional_params = parse_model_name(model_name)

        # Load the model
        model = load_model(model_file, backend, scale_multiplied, additional_params)
        model = model.to(device)

        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders with batch size 1 (for single-sample prediction)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Prediction for both training and validation sets
        for loader, set_type in [(train_loader, 'Train'), (val_loader, 'Validation')]:
            for data in loader:
                if additional_params or scale_multiplied:
                    inputs, params, targets = data
                    inputs, params, targets = inputs.to(device), params.to(device), targets.to(device)
                    outputs = model(inputs, params)
                else:
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                true_weight = targets.item()
                predicted_weight = outputs.item()

                results.append({
                    'model_name': model_name,
                    'true_weight': true_weight,
                    'predicted_weight': predicted_weight,
                    'set_type': set_type,
                    'additional_params': '_'.join(additional_params) if additional_params else None
                })

        # Save results to Feather file after each model
        df_results = pd.concat([df_results, pd.DataFrame(results)], ignore_index=True)
        df_results.reset_index(drop=True, inplace=True)
        feather.write_dataframe(df_results, output_file)
        logger.info(f"Saved results for {model_name}")


if __name__ == "__main__":
    predict_and_save_results()
