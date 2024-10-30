#!/bin/bash
#
#SBATCH --job-name=bincount_ct_scans                            # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:1                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=0-04:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=2G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=4                                 # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/jschoenberg/miniconda/envs/infer_body/


cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/3_Data_Preparation/Justin
echo "Counting bins..."
python Bincount_pixelarray.py

exit