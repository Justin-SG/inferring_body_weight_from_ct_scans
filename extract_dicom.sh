#!/bin/bash
#
#SBATCH --job-name=extract_dicom                            # Job name
#SBATCH --output=extract_dicomt.txt                         # output file
#SBATCH -e extract_dicom.err                                # File to which STDERR will be written
#SBATCH --gres=gpu:1                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=1-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000                                  # Memory in MB per CPU allocated

hostname

conda activate /home2/jschoenberg/miniconda/envs/infer_body/
cd /srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/1_Data_Extraction
python DICOM_to_Dataframe.py -r /srv/GadM/Datasets/AIBA_CT_KG

exit