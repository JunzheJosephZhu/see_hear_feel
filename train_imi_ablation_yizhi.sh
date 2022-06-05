#!/bin/bash
# Usage: sbatch run_slurm.sh
#SBATCH --partition=svl --qos=normal --nodelist=svl8
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
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
source /sailhome/zhangyz/.bashrc
conda activate svl_multi_trans
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd svl_project_
###################
# Run your script #
###################
echo "running command : <RUN_COMMAND>"
# packing
# python svl_project/imi_training/train_transformer.py --exp_name _vg_120_d4 --ablation vg --dim 120 --num_stack 6 --drop_path 0.4
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_120_d4 --ablation vg_t --dim 120 --num_stack 6 --drop_path 0.4
# python svl_project/imi_training/train_transformer.py --exp_name _vg_a_120_d4 --ablation vg_ah --dim 120 --num_stack 6 --drop_path 0.4
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120_d4 --ablation vg_t_ah --dim 120 --num_stack 6 --drop_path 0.4

# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120_d3 --ablation vg_t_ah --dim 120 --drop_path 0.3
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120_d4 --ablation vg_t_ah --dim 120 --drop_path 0.4
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120_d5 --ablation vg_t_ah --dim 120 --drop_path 0.5


# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_96 --ablation vg_t_ah --dim 96 --drop_path 0.2
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_144 --ablation vg_t_ah --dim 144 --drop_path 0.2
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_156 --ablation vg_t_ah --dim 168 --drop_path 0.2

# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_120_s4 --ablation vg_t_ah --dim 120 --num_stack 4
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_120_s6 --ablation vg_t_ah --dim 120 --num_stack 6
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_120_s8 --ablation vg_t_ah --dim 120 --num_stack 8
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_120_s10 --ablation vg_t_ah --dim 120 --num_stack 10
# python svl_project/imi_training/train_transformer.py --exp_name vg_t_a_120_s12 --ablation vg_t_ah --dim 120 --num_stack 12

# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_480_noaux --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_480_noaux_nocrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_240_noaux_nocrop_drop3 --ablation vg_t_ah --dim 240 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --drop_path 0.3
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_240_noaux_nocrop --ablation vg_t_ah --dim 240 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_168_noaux_nocrop --ablation vg_t_ah --dim 168 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop 
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120_noaux_nocrop --ablation vg_t_ah --dim 120 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop 
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_480_noaux_nocrop_drop3 --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --drop_path 0.3

python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600_noaux_nocrop --ablation vg_t_ah --dim 600 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop





