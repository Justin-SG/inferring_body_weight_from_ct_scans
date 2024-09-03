#!/bin/bash
#
#SBATCH --job-name=extract_dicom                            # Job name
#SBATCH --output=%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:1                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=1-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=1000M                                 # Memory in MB per CPU allocated


eval "$(conda shell.bash hook)"
conda activate /home2/jschoenberg/miniconda/envs/infer_body/

# Navigate to the working directory
cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/1_Data_Extraction

# Run the Python script and log output
python DICOM_to_Dataframe.py -r /srv/GadM/Datasets/AIBA_CT_KG/

exit