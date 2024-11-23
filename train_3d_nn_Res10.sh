#!/bin/bash
#
#SBATCH --job-name=res10                                    # Job name
#SBATCH --output=Log/%x_%j.out                              # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                               # Error file (includes job name and ID)
#SBATCH --gres=gpu:2                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=5-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=8                                   # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/phofmann/miniconda/envs/hagrid/


cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/4_Modelling/Patrick

python 3D_CNN.py --epochs=10 --model=resnet_10 --model_depth=10 --pretrained=true --batch_size=1 --learning_rate=0.0001 --patience=5

exit