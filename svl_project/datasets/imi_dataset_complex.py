from argparse import ArgumentParser
from ast import arg
from lib2to3.pytree import Base
from tkinter.messagebox import NO
from grpc import AuthMetadataContext
from importlib_metadata import itertools
from matplotlib.transforms import Transform
import pandas as pd
# from tomlkit import key
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
    def __init__(self, log_file, args, dataset_idx, data_folder="data/test_recordings_0214", train=True):
        super().__init__(log_file, data_folder)
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(
            dataset_idx, load_audio=False)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        xy_space = {-0.0005: 0, 0: 1, 0.0005: 2}
        z_space = {-0.0005: 0, 0: 1, 0.0005: 2}
        r_space = {-0.005: 0, 0: 1, 0.005: 2}
        keyboard = torch.as_tensor(
            [xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]], r_space[keyboard[3]]])
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


class ImitationDatasetWholeSeq(BaseDataset):
    def __init__(self, log_file, args, data_folder="data/test_recordings_0214", train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.fps = 10
        self.sr = 44100
        self.subseq_len = 100
        self.num_cam = args.num_camera
        self.EPS = 1e-8
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width = int(self.resized_width_v * (1.0 - args.crop_percent))

        self.transform_img = T.Compose([
            T.Resize((self.resized_height_v, self.resized_width_v)),
            T.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.2),
            T.RandomCrop((self._crop_height, self._crop_width))
        ])
        self.transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.0),
        ])

        self.use_flow = args.use_flow
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
        return trial, timestamps, len(timestamps["action_history"])

    def __len__(self):
        if self.ablation == 't':
            return self.num_frames - self.approach_end
        else:
            return len(self.logs)

    def __getitem__(self, idx):
        trial, timestamps, num_frames = self.get_episode(idx, load_audio=True)
        # for timestamp in range(num_frames):
        #     img = self.transform(self.load_image("data/test_recordings/2022-03-31 19:32:18.632658/cam_fixed_color/0.png"))
        # i, j, h, w = T.RandomCrop.get_params(img, output_size=(self._crop_height, self._crop_width))
        assert num_frames >= self.subseq_len
        start = torch.randint(low=0, high=num_frames - self.subseq_len, size=())
        end = start + self.subseq_len

        cam_fixed_seq = torch.stack(
            [self.transform_img(self.load_image(trial, "cam_fixed_color", timestep))
                for timestep in range(start, end)], dim=0)
                
        if self.num_cam == 2:
            # [num_frames * 2, H, W]
            cam_gripper_seq = torch.stack(
                [self.resize_image(self.load_image(trial, "cam_gripper_color", timestep), (64, 64))
                    for timestep in range(start, end)], dim=0)

        if not self.use_flow:
            tactile_seq = torch.stack(
                [(
                    self.transform_gel(
                        self.load_image(trial, "left_gelsight_frame", timestep) - self.gelsight_offset)
                 + 0.5).clamp(0, 1) for
                 timestep in range(start, end)], dim=0)
            # cv2.imshow("1",tactile_framestack.cpu().permute(0,2,3,1).numpy()[0,:,:,:])
            # cv2.waitKey(100)
        else:
            tactile_seq = torch.stack([torch.from_numpy(
                        torch.load(os.path.join(trial, "left_gelsight_flow", str(timestep) + ".pt"))).type(
                        torch.FloatTensor)
                    for timestep in range(start, end)], dim=0)
        keyboards = timestamps["action_history"][start:end]
        xy_space = {-0.0005: 0, 0: 1, 0.0005: 2}
        z_space = {-0.0005: 0, 0: 1, 0.0005: 2}
        r_space = {-0.005: 0, 0: 1, 0.005: 2}
        keyboards = torch.as_tensor(
            [[xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]], r_space[keyboard[3]]] for keyboard in keyboards])

        # if self.num_cam == 2:
        #     # [num_frames * 2, 3, H, W]
        #     v_seq = torch.stack((cam_gripper_seq, cam_fixed_seq), dim=1)
        # else:
        v_seq = cam_fixed_seq

        return v_seq, tactile_seq, keyboards


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