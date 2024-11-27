#!/bin/bash
#
#SBATCH --job-name=model_validation                            # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:2                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=5-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=8                                  # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/phofmann/miniconda/envs/hagrid/


cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/4_Modelling/Patrick
echo "Generating Predictions"
python ModelValidation.py
echo "Done"

exit