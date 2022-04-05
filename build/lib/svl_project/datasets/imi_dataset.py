from argparse import ArgumentParser
from tkinter.messagebox import NO
import pandas as pd
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

import numpy as np

EPS = 1e-8

def clip_audio(audio, audio_start, audio_end):
    left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
    if audio_start < 0:
        left_pad = torch.zeros((2, -audio_start))
        audio_start = 0
    if audio_end >= audio.size(-1):
        right_pad = torch.zeros((2, audio_end - audio.size(-1)))
        audio_end = audio.size(-1)
    audio_clip = torch.cat(
        [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
    )
    return audio_clip

class ImitationDataSet_hdf5(IterableDataset):
    def __init__(self, log_file=None, num_stack = 5, frameskip = 2, crop_height = 432, crop_width = 576, data_folder="data/test_recordings_0208_repeat"):
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self._crop_height = crop_height
        self._crop_width = crop_width
        self.num_stack = num_stack
        self.frameskip = frameskip
        self.iter_end = len(self.logs)
        self.idx = 0
        self.load_episode(self.idx)
        self.max_len = (num_stack - 1) * frameskip + 1
        self.framestack_cam_gripper = deque(maxlen=self.max_len)
        self.framestack_cam_fixed = deque(maxlen=self.max_len)
        # self.stack_idx_gripper = deque(maxlen=self.max_len)
        # self.stack_idx_fixed = deque(maxlen=self.max_len)
        self.fps = 10

    def __iter__(self):
        return self

    def __next__(self):
        # get frame
        cam_gripper_idx = next(self.cam_gripper_idx)
        cam_gripper_frame = self.all_datasets['cam_gripper_color'][cam_gripper_idx]
        cam_fixed_idx = next(self.cam_fixed_idx)
        cam_fixed_frame =  self.all_datasets['cam_fixed_color'][cam_fixed_idx]
        if self.cam_gripper_idx == StopIteration or self.cam_fixed_idx == StopIteration:
            self.idx += 1
            if self.idx == len(self.logs):
                self.idx = 0
                self.load_episode(0)
                raise StopIteration
            self.load_episode(self.idx)
        cam_gripper_frame = torch.as_tensor(cam_gripper_frame).permute(2, 0, 1) / 255
        cam_fixed_frame = torch.as_tensor(cam_fixed_frame).permute(2, 0, 1) / 255

        # data augmentation
        transform = T.Compose([
            T.RandomCrop((self._crop_height, self._crop_width)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        cam_gripper_frame = transform(cam_gripper_frame)
        cam_fixed_frame = transform(cam_fixed_frame)

        # stack past frames into the queue
        '''framestack: list [ frame1 (tensor), frame2 (tensor), ... ]'''
        self.framestack_cam_gripper.append(cam_gripper_frame)
        self.framestack_cam_fixed.append(cam_fixed_frame)
        # self.stack_idx_gripper.append(torch.tensor([tuple((idx.start, idx.stop)) for idx in cam_gripper_idx]))
        # self.stack_idx_fixed.append(torch.tensor([tuple((idx.start, idx.stop)) for idx in cam_fixed_idx]))

        # return frames from the stack with frameskip
        if len(self.framestack_cam_gripper) >= self.max_len:
            frameskip_cam_gripper = list(islice(self.framestack_cam_gripper, 0, None, self.frameskip))
            frameskip_cam_fixed = list(islice(self.framestack_cam_fixed, 0, None, self.frameskip))
            # skip_idx_gripper = list(islice(self.stack_idx_gripper, 0, None, self.frameskip))
            # skip_idx_fixed = list(islice(self.stack_idx_fixed, 0, None, self.frameskip))
        else:
            frameskip_cam_gripper = [self.framestack_cam_gripper[-1]] * self.num_stack
            frameskip_cam_fixed = [self.framestack_cam_fixed[-1]] * self.num_stack
            # skip_idx_gripper = [self.stack_idx_gripper[-1]] * self.num_stack
            # skip_idx_fixed = [self.stack_idx_fixed[-1]] * self.num_stack
        
        # processing actions
        action_c = self.timestamps["action_history"][self.timestep]
        xy_space = {-0.003: 0, 0: 1, 0.003: 2}
        z_space = {-0.0015: 0, 0: 1, 0.0015: 2}
        x = xy_space[action_c[0]]
        y = xy_space[action_c[1]]
        z = z_space[action_c[2]]
        action = torch.as_tensor([x, y, z])
        self.timestep += 1
        # print('*' * 50 + f"imi_dataset\nidx_gripper:\n{skip_idx_gripper}\nidx_fixed:\n{skip_idx_fixed}\nidx:\n{skip_idx_gripper + skip_idx_fixed}")
        return frameskip_cam_gripper, frameskip_cam_fixed, action #, skip_idx_gripper + skip_idx_fixed

    def load_episode(self, idx):
        # reset timestep
        self.timestep = 0
        # get file older
        format_time = self.logs.iloc[idx].Time #.replace(":","_")
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        # read camera frames
        self.all_datasets = h5py.File(os.path.join(trial, "data.hdf5"), 'r')
        self.cam_gripper_idx = self.all_datasets['cam_gripper_color'].iter_chunks()
        self.cam_fixed_idx = self.all_datasets['cam_fixed_color'].iter_chunks()


class ImitationDataSet_hdf5_multi(IterableDataset):
    def __init__(self, log_file=None, num_stack=5, frameskip=2, crop_height=432, crop_width=576,
                 data_folder="data/test_recordings_0208_repeat"):
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self._crop_height = crop_height
        self._crop_width = crop_width
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )
        self.num_stack = num_stack
        self.frameskip = frameskip
        self.iter_end = len(self.logs)
        self.idx = 0
        self.load_episode(self.idx)
        self.framestack_cam = deque(maxlen=num_stack * frameskip)
        self.framestack_fixed_cam = deque(maxlen=num_stack * frameskip)
        self.framestack_gel = deque(maxlen=num_stack * frameskip)
        self.max_len = num_stack * frameskip
        self.fps = 10

    def __iter__(self):
        return self

    def __next__(self):
        # start = time.time()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert worker_info.num_workers == 1, "multiworker not supported"
        sr = int(16000 * (self.fps / (self.frameskip * self.num_stack)))
        resolution = sr // 10

        cam_frame = self.all_datasets['cam_gripper_color'][next(self.cam_gripper_idx)]
        cam_fixed_frame = self.all_datasets['cam_fixed_color'][next(self.cam_fix_idx)]
        gs_frame = self.all_datasets['left_gelsight_flow'][next(self.gs_idx)]
        if self.cam_gripper_idx == StopIteration:
            # self.gs_video.release()
            self.idx += 1
            if self.idx == len(self.logs):
                self.idx = 0
                self.load_episode(0)
                raise StopIteration
            self.load_episode(self.idx)


        cam_frame = torch.as_tensor(cam_frame).permute(2, 0, 1) / 255
        cam_fixed_frame = torch.as_tensor(cam_fixed_frame).permute(2, 0, 1) / 255

        transform = T.Compose([
            T.RandomCrop((self._crop_height, self._crop_width)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

        cam_frame = transform(cam_frame)
        cam_fixed_frame = transform(cam_fixed_frame)

        self.framestack_cam.append(cam_frame)
        self.framestack_fixed_cam.append(cam_fixed_frame)
        self.framestack_gel.append(gs_frame)

        if len(self.framestack_cam) >= self.max_len:
            cam_framestack = torch.vstack(tuple(islice(self.framestack_cam, 0, None, self.frameskip)))
            cam_fixed_framestack = torch.vstack(tuple(islice(self.framestack_fixed_cam, 0, None, self.frameskip)))
            gs_framestack = torch.vstack(tuple(islice(self.framestack_gel, 0, None, self.frameskip)))
        else:
            cam_framestack = torch.vstack(([self.framestack_cam[-1]] * self.num_stack))
            cam_fixed_framestack = torch.vstack(([self.framestack_fixed_cam[-1]] * self.num_stack))
            gs_framestack = torch.vstack(([self.framestack_gel[-1]] * self.num_stack))

        # load audio clip
        # audio length is 1 second
        audio_start = self.timestep * resolution - sr // 2
        audio_end = audio_start + sr
        audio_clip = clip_audio(self.audio, audio_start, audio_end)
        assert audio_clip.size(1) == sr
        spec = self.mel(audio_clip)
        log_spec = torch.log(spec + EPS)
        action_c = self.timestamps["action_history"][self.timestep]
        xy_space = {-0.006: 0, 0: 1, 0.006: 2}
        z_space = {-0.003: 0, 0: 1, 0.003: 2}
        x = xy_space[action_c[0]]
        y = xy_space[action_c[1]]
        z = z_space[action_c[2]]
        action = torch.as_tensor([x, y, z])
        self.timestep += 1

        return cam_framestack, cam_fixed_framestack, gs_framestack, log_spec, action

    def load_episode(self, idx):
        # reset timestep
        self.timestep = 0
        # get file older
        format_time = self.logs.iloc[idx].Time.replace(":", "_")
        trial = os.path.join(self.data_folder, format_time)
        # load audio tracks
        self.audio = torch.as_tensor(np.stack([self.all_datasets['audio_gripper'], self.all_datasets['audio_holebase']], 0)).float()
        # load json file
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        # read all data
        self.all_datasets = h5py.File(os.path.join(trial, "data.hdf5"), 'r')
        self.cam_gripper_idx = self.all_datasets['cam_gripper_color'].iter_chunks()
        self.cam_fix_idx = self.all_datasets['cam_fixed_color'].iter_chunks()
        self.gs_idx = self.all_datasets['left_gelsight_flow'].iter_chunks()



if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train_0331.csv")
    parser.add_argument("--num_stack", default=5, type=int)
    parser.add_argument("--frameskip", default=2, type=int)
    parser.add_argument("--data_folder", default="data/test_recordings_0208_repeat")
    args = parser.parse_args()

    dataset = ImitationDataSet_hdf5("train_0331.csv")
    # print("dataset", dataset.len)
    cnt = 0
    for cam_frame, _, _ in dataset:
        cnt+=1
        if cnt >=5:
            break