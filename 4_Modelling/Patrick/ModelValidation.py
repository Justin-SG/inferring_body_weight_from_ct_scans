import sys
import pandas as pd
import torch
from pathlib import Path
import logging
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import feather
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, ViT_B_16_Weights, ViT_B_32_Weights
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adding Project Paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model' / '3D'
eval_dir = project_dir / '5_Evaluation' / 'Patrick'

sys.path.append(str(project_dir / '3_Data_Preparation'))
sys.path.append(str(project_dir / '4_Modelling/Patrick/medicalNet'))

from CT_Datasets import CtScanDataset
from Transforms import Transforms
from CustomModels import ResNetRegression
from setting import Options
from model import generate_model

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the datasets
query = 'BodyPart == "Stamm"'
dataset = CtScanDataset(query, transform=Transforms.CNN3DPreprocessor())


# Function to parse model name and select the appropriate dataset
def parse_model_name(model_name):
    parts = model_name.replace(".pth", "").split("_")
    backend = parts[0]

    # Extract additional params
    model_depth = int(parts[1])

    return backend, model_depth


def get_model(model_depth):
    """Load the pretraind model."""
    # settting
    sets = Options()
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.model_depth = model_depth

    net, _ = generate_model(sets)

    # Instantiate the new regression model
    model = ResNetRegression(net, num_outputs=1)

    return model


# Function to load model
def load_model(model_path, model_depth):
    model = get_model(model_depth)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Prediction loop
def predict_and_save_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # Get all model files
    model_files = list(model_dir.glob("*best.pth"))

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
        backend, model_depth = parse_model_name(model_name)

        # Load the model
        model = load_model(model_file, model_depth)
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
            for i, data in enumerate(loader):
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
                    'patient_id': dataset.dicom_df.loc[loader.dataset.indices[i]].PatientId,
                    'pixel_array_file': dataset.dicom_df.loc[loader.dataset.indices[i]].PixelArrayFile
                })

        # Save results to Feather file after each model
        df_results = pd.concat([df_results, pd.DataFrame(results)], ignore_index=True)
        df_results.reset_index(drop=True, inplace=True)
        feather.write_dataframe(df_results, output_file)
        logger.info(f"Saved results for {model_name}")


if __name__ == "__main__":
    predict_and_save_results()
