import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import torch
import numpy as np
# from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti
from svl_project.datasets.imi_dataset_complex import ImitationDatasetFramestackMulti
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_tactile_flow_encoder, make_audio_encoder
from svl_project.models.imi_models import Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T
from torch.autograd import Variable
import shutil
from tqdm import tqdm


class MakeVideo():
    def __init__(self, dir=None, framestack=None, name='default', num_episode=1, mods='v_t', args=None, length=None):
        self.dir = dir
        self.framestack = framestack
        self.name = name
        self.save_dir = os.path.join(self.dir, self.name)
        self.action_dim = args.action_dim
        self.length = length
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)
        self.mydict = {
            'v': {
                'resolution': (160, 120),
                # 'resolution': (160, 120),
                'font_scale': .5,
                'thick_scale': 1},
            't': {'resolution': (100, 100),
                  'font_scale': .7,
                  'thick_scale': 1},
        }
        self.path_dict = {'v': [],
                          't': []}
        self.video_writer = {}
        self.frames_ = []

    def initialize_sep(self):
        ## save the histogram and action sequence
        self.subdirs = {}
        for dirkey in ['hist', 'seq', 'arrow']:
            self.subdirs[dirkey] = os.path.join(self.save_dir, dirkey)
            if os.path.exists(self.subdirs[dirkey]):
                shutil.rmtree(self.subdirs[dirkey])
            os.mkdir(self.subdirs[dirkey])
        # initialize image for action sequence
        self.fig_seq, self.axs_seq = plt.subplots(self.action_dim + 1, 1, figsize=(6, 4), sharex='col')
        self.fig_seq.subplots_adjust(hspace=.4)
        for i in range(0, self.action_dim + 1):
            if i > 0:
                self.axs_seq[i].set_ylim([-1.5, 1.5])
            self.axs_seq[i].set_xlim([0, self.length])

        ## save observations
        for i in range(self.framestack):
            v_path = os.path.join(self.save_dir, f"v{i}" + '.avi')
            self.video_writer[f"v{i}"] = cv2.VideoWriter(
                v_path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10,
                self.mydict['v']['resolution']
            )
            t_path = os.path.join(self.save_dir, f"t{i}" + '.avi')
            self.video_writer[f"t{i}"] = cv2.VideoWriter(
                t_path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10,
                self.mydict['t']['resolution']
            )
            self.path_dict['v'].append(v_path)
            self.path_dict['t'].append(t_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = 255, 50, 50
        self.thickness = cv2.LINE_8

    def save_obs(self, imgs, item, pred=None, gt=None, step=None):
        if item in ['hist', 'seq', 'arrow']:
            imgs.savefig(os.path.join(self.subdirs[item], f"{step}.png"))
            return
        for i in range(self.framestack):
            img = imgs[i]
            if item == 'v':
                img = imgs[i].cpu().permute(1, 2, 0).numpy()
            elif item == 't':
                img = imgs[i].cpu().permute(2, 1, 0).numpy()
            img = np.uint8(img * 255)
            if item == 'v':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img,
                            str(i),
                            (5, 20),
                            self.font,
                            self.mydict[item]['font_scale'],
                            self.color,
                            self.mydict[item]['thick_scale'],
                            self.thickness
                            )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.video_writer[item + str(i)].write(img)
