#!/bin/bash
# Usage: sbatch run_slurm.sh
#SBATCH --partition=svl --qos=normal --nodelist=svl15
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --job-name="immi"
#SBATCH --output=logs/immi_slurm_%A.out
#SBATCH --error=logs/immi_slurm_%A.err

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
source /sailhome/li2053/.bashrc
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
# note that vg represents third view cam, vf represents gripper view
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_ah_t --ablation vg_ah_t
python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _ah --ablation ah
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _t --ablation t
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _ah_t --ablation ah_t

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vf_ft --ablation vf_ft
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_ft --ablation vg_ft
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_vf --ablation vg_vf
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_vf_ft --ablation vg_vf_ft



# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg --ablation vg
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t --ablation vg_t 
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_a --ablation vg_ah 

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a --ablation vg_t_ah 

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_m --ablation vg --use_mha 
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_m --ablation vg_t --use_mha 
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_a_m --ablation vg_ah --use_mha
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m --ablation vg_t_ah --use_mha
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m_q --ablation vg_t_ah --use_mha --use_query --encoder_dim 512

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m --ablation vg_t_ah --use_mha --norm_audio
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m --ablation vg_t_ah --use_mha --norm_freq

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_l --ablation vg_t_ah --use_lstm 

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_no_t --ablation vg_t_ah --use_mha --remove_temp
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_no_m --ablation vg_t_ah --use_mha --remove_modal


# pour
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_20 --ablation vg_t_ah --num_stack 20

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_m_r --ablation vg --use_mha --num_stack 20
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_m_r --ablation vg_t --use_mha --num_stack 20
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_a_m_r --ablation vg_ah --use_mha --num_stack 20
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m_r --ablation vg_t_ah --use_mha
# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_m_nr --ablation vg_t_ah --use_mha --num_stack 20

# python svl_project/imi_training/train_imitation.py --config conf/imi/imi_learn.yaml --exp_name _vg_t_a_l --ablation vg_t_ah --use_lstm









