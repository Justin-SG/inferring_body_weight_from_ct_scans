#!/bin/bash
#
#SBATCH --job-name=train_2d_nn                            # Job name
#SBATCH --output=%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:4                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=4-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=10                                  # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/jschoenberg/miniconda/envs/infer_body/


cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/4_Modelling/Flo

echo "Training 2D Scan Neural Networks"
python 2D_CNN_Projection.py --epochs=20 --batch_size=32 --lr=0.001 --patience=5

exit