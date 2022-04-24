from argparse import ArgumentParser
from ast import arg
from lib2to3.pytree import Base
from tkinter.messagebox import NO
from grpc import AuthMetadataContext
from importlib_metadata import itertools
from matplotlib.transforms import Transform
import pandas as pd
from tomlkit import key
from torch.utils.data import Dataset, IterableDataset
import os
import soundfile as sf
import torch
import h5py
import torchvision.transforms as T
from itertools import islice

import json
import cv2
import time
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
import torchaudio
import math
from re import L
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from svl_project.datasets.base import BaseDataset
import numpy as np
import random

from PIL import Image, ImageEnhance, ImageOps

EPS = 1e-8


def augment_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image

class ImitationDatasetLabelCount(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder=None, train=True):
        super().__init__(log_file, data_folder)
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(
            dataset_idx, load_audio=False)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        x_space = {-.0004: 0, 0: 1, .0004: 2}
        y_space = {-.0002: 0, 0: 1, .0002: 2}
        # z_space = {-.0005: 0, 0: 1, .0005: 2}
        # r_space = {-.005: 0, 0: 1, .005: 2}
        # keyboard = torch.as_tensor(
        #     [xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]], r_space[keyboard[3]]])
        z_space = {-.0002: 0, 0: 1, .0002: 2}
        keyboard = torch.as_tensor(
            [x_space[keyboard[0]], y_space[keyboard[1]], z_space[keyboard[2]]])
        return keyboard

    def get_episode(self, idx, load_audio=False):
        """
        Return:
            folder for trial
            logs
            audio tracks
            number of frames in episode
        """
        format_time = self.logs.iloc[idx].Time#.replace(":", "_")
        # print("override" + '#' * 50)
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)
        if load_audio:
            audio_gripper_left = sf.read(os.path.join(trial, 'audio_gripper_left.wav'))[0]
            audio_gripper_right = sf.read(os.path.join(trial, 'audio_gripper_right.wav'))[0]
            audio_holebase_left = sf.read(os.path.join(trial, 'audio_holebase_right.wav'))[0]
            audio_holebase_right = sf.read(os.path.join(trial, 'audio_holebase_right.wav'))[0]
            audio_gripper = torch.as_tensor(np.stack([audio_gripper_left, audio_gripper_right], 0))
            audio_holebase = torch.as_tensor(np.stack([audio_holebase_left, audio_holebase_right], 0))
        else:
            audio_gripper = None
            audio_holebase =None
        return trial, timestamps, audio_gripper, audio_holebase, len(timestamps["action_history"])

class ImitationDatasetFramestackMulti(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder="data/test_recordings_0214", train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip + 1
        self.fps = 10
        self.sr = 44100
        self.resolution = self.sr // self.fps  # number of audio samples in one image idx
        self.audio_len = int(self.resolution * (max(self.max_len, 10)))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=int(self.sr * 0.025), hop_length=int(self.sr * 0.01), n_mels=64
        )
        self.num_cam = args.num_camera
        self.EPS = 1e-8
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(dataset_idx, load_audio=True)
        
        # self.approach_end = self.timestamps['approach_end_idx'][0]
        
        if not args.use_holebase:
            self.audio = self.audio_gripper
        else:
            self.audio = self.audio_holebase[1].unsqueeze(0)
        self.use_flow = args.use_flow
        ## saving initial gelsight frame
        # self.static_gs = self.load_image(os.path.join(self.data_folder, 'static_gs'), "left_gelsight_frame", 0)
        # self.static_gs = self.load_image(self.trial, "left_gelsight_frame", 0)
        # saving the offset
        self.gelsight_offset = torch.as_tensor(
            np.array(Image.open(os.path.join(self.data_folder, 'gs_offset.png')))).float().permute(2, 0,
                                                                                                   1) / 255
        self.ablation = args.ablation
        self.action_dim = args.action_dim
        

    def get_episode(self, idx, load_audio=True):
        """
        Return:
            folder for trial
            logs
            audio tracks
            number of frames in episode
        """
        format_time = self.logs.iloc[idx].Time#.replace(":", "_")
        # print("override" + '#' * 50)
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)
        if load_audio:
            audio_gripper_left = sf.read(os.path.join(trial, 'audio_gripper_left.wav'))[0]
            audio_gripper_right = sf.read(os.path.join(trial, 'audio_gripper_right.wav'))[0]
            audio_holebase_left = sf.read(os.path.join(trial, 'audio_holebase_right.wav'))[0]
            audio_holebase_right = sf.read(os.path.join(trial, 'audio_holebase_right.wav'))[0]
            audio_gripper = torch.as_tensor(np.stack([audio_gripper_left, audio_gripper_right], 0))
            audio_holebase = torch.as_tensor(np.stack([audio_holebase_left, audio_holebase_right], 0))
        else:
            audio_gripper = None
            audio_holebase = None
        return trial, timestamps, audio_gripper, audio_holebase, len(timestamps["action_history"])

    def __len__(self):
        # if self.ablation == 't' or True:
        #     return self.num_frames - self.approach_end
        # else:
        return self.num_frames

    def __getitem__(self, idx):
        # idx = idx + self.approach_end
        # if idx < self.num_frames / 2 and (self.ablation == 't' or self.ablation == 'a'):
        #     print("only use data that contact the surface")
        #     return self.__getitem__(torch.randint(low = int(self.num_frames/2), high=int(self.num_frames),size=()).numpy())
        end = idx  # torch.randint(high=num_frames, size=()).item()
        start = end - self.max_len
        if start < 0:
            cam_idx = [end] * self.num_stack
        else:
            cam_idx = list(np.arange(start + 1, end + 1, self.frameskip))

        if self.train:
            # load camera frames
            transform = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.2),
            ])
            img = transform(self.load_image(self.trial, "cam_fixed_color", end))
            i_v, j_v, h_v, w_v = T.RandomCrop.get_params(img, output_size=(self._crop_height_v, self._crop_width_v))

            transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                # T.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.0),
            ])

            img_g = transform(self.load_image(self.trial, "left_gelsight_frame", end))
            i_t, j_t, h_t, w_t = T.RandomCrop.get_params(img_g, output_size=(self._crop_height_t, self._crop_width_t))

            if self.num_cam == 2:
                cam_gripper_framestack = torch.stack(
                    [T.functional.crop(transform(self.load_image(self.trial, "cam_gripper_color", timestep)), i_v, j_v, h_v,
                                       w_v)
                     for timestep in cam_idx], dim=0)

            cam_fixed_framestack = torch.stack(
                [T.functional.crop(transform(self.load_image(self.trial, "cam_fixed_color", timestep)), i_v, j_v, h_v, w_v)
                 for timestep in cam_idx], dim=0)

            if not self.use_flow:
                tactile_framestack = torch.stack(
                    [T.functional.crop((transform_gel(
                        self.load_image(self.trial, "left_gelsight_frame", timestep)
                        ## input difference between current frame and initial (static) frame instead of the frame itself
                        - self.gelsight_offset
                    ) + 0.5).clamp(0, 1), i_t, j_t, h_t, w_t)  for
                    timestep in cam_idx], dim=0)
                # cv2.imshow("1",tactile_framestack.cpu().permute(0,2,3,1).numpy()[0,:,:,:])
                # cv2.waitKey(100)
            else:
                tactile_framestack = torch.stack(
                    [torch.from_numpy(
                        torch.load(os.path.join(self.trial, "left_gelsight_flow", str(timestep) + ".pt"))).type(
                        torch.FloatTensor)
                    for timestep in cam_idx], dim=0)

        else:
            # load camera frames
            transform = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.CenterCrop((self.resized_height_v, self.resized_width_v))
            ])

            transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.CenterCrop((self.resized_height_t, self.resized_width_t))
            ])

            if self.num_cam == 2:
                cam_gripper_framestack = torch.stack(
                    [transform(self.load_image(self.trial, "cam_gripper_color", timestep))
                     for timestep in cam_idx], dim=0)

            cam_fixed_framestack = torch.stack(
                [transform(self.load_image(self.trial, "cam_fixed_color", timestep))
                 for timestep in cam_idx], dim=0)

            if not self.use_flow:
                tactile_framestack = torch.stack(
                    [(transform_gel(
                        self.load_image(self.trial, "left_gelsight_frame", timestep)
                        ## input difference between current frame and initial (static) frame instead of the frame itself
                        - self.gelsight_offset
                    ) + 0.5).clamp(0, 1) for
                    timestep in cam_idx], dim=0)
                # cv2.imshow("1",tactile_framestack.cpu().permute(0,2,3,1).numpy()[0,:,:,:])
                # cv2.waitKey(100)
            else:
                tactile_framestack = torch.stack(
                    [torch.from_numpy(
                        torch.load(os.path.join(self.trial, "left_gelsight_flow", str(timestep) + ".pt"))).type(
                        torch.FloatTensor)
                    for timestep in cam_idx], dim=0)

        # load audio
        audio_end = end * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        audio_clip = self.clip_audio(self.audio, audio_start, audio_end)
        spec = self.mel(audio_clip.type(torch.FloatTensor))
        log_spec = torch.log(spec + EPS)

        keyboard = self.timestamps["action_history"][end]
        x_space = {-.0004: 0, 0: 1, .0004: 2}
        y_space = {-.0002: 0, 0: 1, .0002: 2}
        # z_space = {-.0005: 0, 0: 1, .0005: 2}
        # r_space = {-.005: 0, 0: 1, .005: 2}
        # keyboard = torch.as_tensor(
        #     [xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]], r_space[keyboard[3]]])
        z_space = {-.0002: 0, 0: 1, .0002: 2}
        keyboard = torch.as_tensor(
            [x_space[keyboard[0]], y_space[keyboard[1]], z_space[keyboard[2]]])

        if self.num_cam == 2:
            v_framestack = torch.cat((cam_gripper_framestack, cam_fixed_framestack), dim=0)
        else:
            v_framestack = cam_fixed_framestack

        # if self.action_dim == 4:
        #     x = keyboard[0] * 27 + keyboard[1] * 9 + keyboard[2] * 3 + keyboard[3]
        # elif self.action_dim == 3:
        #     x = keyboard[0] * 9 + keyboard[1] * 3 + keyboard[2]
        # print(x)

        return v_framestack, tactile_framestack, log_spec, keyboard

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train.csv")
    parser.add_argument("--num_stack", default=5, type=int)
    parser.add_argument("--frameskip", default=2, type=int)
    parser.add_argument("--data_folder", default="data/test_recordings_0220_toy")
    args = parser.parse_args()

    dataset = ImitationDatasetFramestackMulti("train.csv")
    # print("dataset", dataset.len)
    cnt = 0
    zero_cnt = 0
    t_l = []
    num_frame = 0
    for _ in range(11800):
        index = torch.randint(high=10, size=()).item()
        _, _, _, idx, t, num = dataset.__getitem__(index)
        if idx == 0:
            num_frame = num
            zero_cnt += 1
            t_l.append(t)
        cnt += 1
    mydic = {i: t_l.count(i) for i in t_l}
    print(zero_cnt)
    print(num_frame)
    print(len(mydic))

    print(mydic)