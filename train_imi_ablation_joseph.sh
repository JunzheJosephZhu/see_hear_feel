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
# packing
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_192 --ablation vg_t_ah --dim 192 --depth 12
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_192 --ablation vg_t_ah --dim 192 --depth 8
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_192 --ablation vg_t_ah --dim 192 --depth 6
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_192 --ablation vg_t_ah --dim 192 --depth 10

# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_120 --ablation vg_t_ah --dim 120 --depth 12
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_252 --ablation vg_t_ah --dim 252 --depth 12
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_312 --ablation vg_t_ah --dim 312 --depth 12
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_432 --ablation vg_t_ah --dim 432 --depth 12


# Joseph's experiments
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_768 --ablation vg_t_ah --dim 768 --depth 12 --drop_path 0.3
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_768_noaux --ablation vg_t_ah --dim 768 --depth 12 --drop_path 0.3 --aux_multiplier 0.0
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600 --ablation vg_t_ah --dim 600 --depth 12 --drop_path 0.3
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600_nodrop --ablation vg_t_ah --dim 600 --depth 12 --drop_path 0.1
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600_zerodrop --ablation vg_t_ah --dim 600 --depth 12 --drop_path 0.0 --batch_size 8
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600_highlr --ablation vg_t_ah --dim 600 --depth 12 --drop_path 0.3 --lr 0.0005 --batch_size 8
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_960 --ablation vg_t_ah --dim 960 --depth 12 --drop_path 0.3 --batch_size 4
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_432_period1 --ablation vg_t_ah --dim 432 --depth 12 --period 1 --batch_size 16
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_432_period1_depth16 --ablation vg_t_ah --dim 432 --depth 16 --period 1 --batch_size 12




# python svl_project/imi_training/train_transformer.py --exp_name _spec_480_noaux_nocrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_480_noaux_hascrop --ablation vg_t_ah --dim 480 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_192_noaux_nocrop --ablation vg_t_ah --dim 192 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop
# python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_600_noaux_nocrop --ablation vg_t_ah --dim 600 --depth 12 --period 1 --batch_size 16 --aux_multiplier 0.0 --nocrop --task final_para_search
python svl_project/imi_training/train_transformer.py --exp_name _vg_t_a_768_noaux_nocrop --ablation vg_t_ah --dim 768 --depth 12 --period 1 --batch_size 8 --aux_multiplier 0.0 --nocrop --task final_para_search



