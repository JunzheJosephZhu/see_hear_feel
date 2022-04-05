#!/bin/sh

python ./svl_project/imi_training/train_imitation_Ablation.py --ablation v --epochs 1 --num_episode 1

python ./svl_project/imi_training/train_imitation_Ablation.py --ablation t --epochs 1 --num_episode 1
