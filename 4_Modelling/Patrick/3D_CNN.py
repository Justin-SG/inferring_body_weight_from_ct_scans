import os
import sys
import argparse
import logging
from pathlib import Path
from fastai.vision.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from fastai.callback.progress import ProgressCallback
from fastai.metrics import mae  # mean absolute error
from torch.utils.data import random_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # Directs logs to stdout
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

# Define the project paths
project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model/3D'
plot_dir = model_dir / "plots"
stats_dir = model_dir / "stats"
plot_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)

sys.path.append(str(project_dir / '3_Data_Preparation'))
sys.path.append(str(project_dir / '4_Modelling/Patrick/medicalNet'))

# Import project-specific modules
from CT_Datasets import CtScanDataset
from Transforms import Transforms
from CustomModels import ResNetRegression
from setting import Options
from model import generate_model
from logger import log

# Argument Parser
parser = argparse.ArgumentParser(description="Train ResNet or Vision Transformer on hand gesture dataset.")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for training")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping")
parser.add_argument('--model', choices=['resnet_10', 'resnet_18'], default='resnet_10', help="Model type to train")

args = parser.parse_args()

# Load Dataset
def get_dataloaders(query, batch_size, transforms, num_workers=2):
    logger.info("Loading datasets...")
    dataset = CtScanDataset(query, transform=transforms, return_tensor=True)

    train_dataset, val_dataset = random_split(
        dataset,
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Datasets loaded successfully.")
    return train_dl, val_dl

# Define Model
def get_model(model_type):
    logger.info(f"Initializing model: {model_type}")
    # settting
    sets = Options()
    sets.target_type = "normal"
    sets.phase = 'test'

    sets.resume_path = sets.pretrain_path = model_dir / f"{model_type}.pth"
    #checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)

    # Instantiate the new regression model
    model = ResNetRegression(net, num_outputs=1)

    # If model already exists, load it
    if (model_dir / f"{model_type}_best.pth").exists():
        logger.info("Loading existing model...")
        model.load_state_dict(torch.load(model_dir / f"{model_type}_best.pth", weights_only=True))

    logger.info("Model initialized successfully.")
    return model

def save_loss_curve(learn, model_type):
    logger.info("Saving loss curve plot...")
    fig, ax = plt.subplots()
    loss_curve_filename = plot_dir / f"{model_type}_loss_curve.png"
    ax = learn.recorder.plot_loss(show_epochs=True, with_valid=True)
    plt.savefig(loss_curve_filename)
    plt.close()
    logger.info(f"Loss curve plot saved at {loss_curve_filename}")

# Train Model
def train_model(query, model_type, epochs, batch_size, learning_rate, patience):
    model = get_model(model_type)
    transforms = Transforms.CNN3DPreprocessor()
    dataloaders = get_dataloaders(query, batch_size, transforms)
    train_dl, val_dl = dataloaders

    logger.info("Starting training process...")
    learn = Learner(
        dls=DataLoaders(train_dl, val_dl),
        model=model,
        loss_func=L1LossFlat(),
        metrics=[mae],
        cbs=[
            ProgressCallback(),
            CSVLogger(fname=stats_dir / f"{model_type}_training_logs.csv", append=True),
            EarlyStoppingCallback(monitor='valid_loss', patience=patience),
            SaveModelCallback(monitor='valid_loss', fname=model_dir / f"{model_type}_best")
        ]
    )

    model_summary_filename = stats_dir / f"{model_type}_model_summary.txt"
    with open(model_summary_filename, 'w') as f:
        f.write(learn.summary())
    logger.info(f"Model summary saved at {model_summary_filename}")

    with learn.no_bar():
        learn.fit_one_cycle(epochs, learning_rate)

    # Save the loss curve for tracking progress
    save_loss_curve(learn, model_type)
    logger.info("Training complete.")
    return learn

# Main Execution
if __name__ == "__main__":
    # Define the datasets
    query = 'BodyPart == "Stamm"'

    learn = train_model(query, args.model, args.epochs, args.batch_size, args.learning_rate, args.patience)
    logger.info("Process completed successfully.")
