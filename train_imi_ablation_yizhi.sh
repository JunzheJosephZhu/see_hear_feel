#!/bin/bash
# Usage: sbatch run_slurm.sh
#SBATCH --partition=svl --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
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
source /sailhome/zhangyz/.bashrc
conda activate svl_multi
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd svl_fs
###################
# Run your script #
###################
echo "running command : <RUN_COMMAND>"
# packing
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a --ablation vg_t_ah --train_csv train_0603.csv --val_csv val_0603.csv

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_m --ablation vg --use_mha --train_csv train_0603.csv --val_csv val_0603.csv
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_m --ablation vg_t --use_mha --train_csv train_0603.csv --val_csv val_0603.csv
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_a_m --ablation vg_ah --use_mha --train_csv train_0603.csv --val_csv val_0603.csv
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m --ablation vg_t_ah --use_mha --train_csv train_0603.csv --val_csv val_0603.csv

python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_l --ablation vg_t_ah --use_lstm --train_csv train_0603.csv --val_csv val_0603.csv