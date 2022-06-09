#!/bin/bash
# Usage: sbatch run_slurm.sh
#SBATCH --partition=svl --qos=normal 
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:4
#SBATCH --job-name="immi"
#SBATCH --output=/viscam/u/li2053/logs/immi_slurm_%A.out
#SBATCH --error=/viscam/u/li2053/logs/immi_slurm_%A.err

######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
source /sailhome/josef/.bashrc
conda activate joseph
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /viscam/u/li2053/svl_project_
###################
# Run your script #
###################
echo "running command : <RUN_COMMAND>"
# pouring
# python svl_project/imi_training/train_transformer.py --exp_name pouring_480_noaux_nocrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 12 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring
# python svl_project/imi_training/train_transformer.py --exp_name pouring_360_noaux_nocrop --ablation vg_t_ah --dim 360 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring
# python svl_project/imi_training/train_transformer.py --exp_name pouring_600_noaux_nocrop --ablation vg_t_ah --dim 600 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring
# python svl_project/imi_training/train_transformer.py --exp_name pouring_480_noaux_hascrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring
# python svl_project/imi_training/train_transformer.py --exp_name pouring_1dconv_480_noaux_nocrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 12 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring --use_1dconv




python svl_project/imi_training/train_transformer.py --exp_name pouring_480_noaux_nocrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 12 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring

# python svl_project/imi_training/train_transformer.py --exp_name pouring_192_noaux_nocrop --ablation vg_t_ah --dim 192 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring

# python svl_project/imi_training/train_transformer.py --exp_name pouring_600_noaux_nocrop --ablation vg_t_ah --dim 600 --depth 12 --period 1 --batch_size 8 --aux_multiplier 0.0 --nocrop --action_dim 2 --train_csv train_0605.csv --val_csv val_0605.csv --data_folder data/data_0605/test_recordings --task pouring

