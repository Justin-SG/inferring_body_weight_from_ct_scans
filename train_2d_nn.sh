#!/bin/bash
#
#SBATCH --job-name=train_2d_nn                            # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:2                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=5-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=8                                  # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/jschoenberg/miniconda/envs/infer_body/


cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/4_Modelling/Flo

python 2D_Projection_NN.py --epochs=20 --batch_size=64 --learning_rate=0.0001 --patience=5

exit