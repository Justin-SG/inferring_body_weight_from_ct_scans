import sys
import pandas as pd
import torch
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

sys.path.append(str(project_dir / '3_Data_Preparation'))

# TODO