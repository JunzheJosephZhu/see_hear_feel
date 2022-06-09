import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import yaml
import shutil
from tqdm import tqdm


class MakeVideo():
    def __init__(self, dir=None, framestack=None, name='default', num_episode=1, mods='v_t', args=None, length=None):
        self.dir = dir
        self.framestack = framestack
        self.name = name
        if self.name is None:
            self.save_dir = self.dir
        else:
            self.save_dir = os.path.join(self.dir, self.name)
        if args.pouring:
            self.action_dim = args.action_dim
        else:
            self.action_dim = 3
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
            't': {'resolution': (75, 100),
                #   'resolution': (100, 100),
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
        for dirkey in ['hist', 'seq', 'arrow', 'audio', 'confusion']:
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
        if item in ['hist', 'seq', 'arrow', '   ', 'audio', 'confusion']:
            if item == 'audio':
                log_spec = imgs[0]
                audio_clip = imgs[1]
                spec_nomel = torch.fft.rfft(audio_clip.type(torch.FloatTensor))
                # print(spec_nomel.shape)
                # print(audio_clip.shape)
                
                # plt.figure(0)
                fig_audio, arrs_audio = plt.subplots(2, figsize=(6, 3))
                # print(x)
                ## plot mel spec
                arrs_audio[0].imshow(log_spec[0][0])
                ## plot rfft
                x = torch.fft.rfftfreq(len(audio_clip[0][0]), 1 / 44100)
                arrs_audio[1].plot(x[:10000], np.abs(spec_nomel[0][0])[:10000])
                fig_audio.tight_layout()
                
                imgs = fig_audio

            imgs.savefig(os.path.join(self.subdirs[item], f"{step}.png"))
            plt.close(imgs)
            return
        for i in range(self.framestack):
            img = imgs[i]
            if item == 'v':
                img = imgs[i].cpu().permute(1, 2, 0).numpy()
            elif item == 't':
                # just the above tactile resolution
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
