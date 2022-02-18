import itertools
from argparse import ArgumentParser
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import os
import soundfile as sf
import torch
import h5py
import torchvision.transforms as T

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

class ImmitationDataSet_hdf5(IterableDataset):
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
        self.framestack_cam = deque(maxlen = (num_stack-1) * frameskip + 1)
        self.framestack_fixed_cam = deque(maxlen = (num_stack-1) * frameskip + 1)
        self.max_len = (num_stack-1) * frameskip + 1
        self.fps = 10

    def __iter__(self):
        return self

    def __next__(self):

        cam_frame = self.all_datasets['cam_gripper_color'][next(self.cam_gripper_idx)]
        cam_fixed_frame =  self.all_datasets['cam_fixed_color'][next(self.cam_fix_idx)]
        if self.cam_gripper_idx == StopIteration or self.cam_fix_idx == StopIteration:
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

        if len(self.framestack_cam) >= self.max_len:
            cam_framestack = torch.vstack(tuple(itertools.islice(self.framestack_cam, 0, None, self.frameskip)))
            cam_fixed_framestack = torch.vstack(tuple(itertools.islice(self.framestack_cam, 0, None, self.frameskip)))
        else:
            cam_framestack = torch.vstack(([self.framestack_cam[-1]] * self.num_stack))
            cam_fixed_framestack = torch.vstack(([self.framestack_cam[-1]] * self.num_stack))

        action_c = self.timestamps["action_history"][self.timestep]
        xy_space = {-0.003: 0, 0: 1, 0.003: 2}
        z_space = {-0.0015: 0, 0: 1, 0.0015: 2}
        x = xy_space[action_c[0]]
        y = xy_space[action_c[1]]
        z = z_space[action_c[2]]
        action = torch.as_tensor([x, y, z])
        self.timestep += 1
        return cam_framestack, cam_fixed_framestack, action

    def load_episode(self, idx):
        # reset timestep
        self.timestep = 0
        # get file older
        format_time = self.logs.iloc[idx].Time.replace(":","_")
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        # read camera frames
        self.all_datasets = h5py.File(os.path.join(trial, "data.hdf5"), 'r')
        self.cam_gripper_idx = self.all_datasets['cam_gripper_color'].iter_chunks()
        self.cam_fix_idx = self.all_datasets['cam_fixed_color'].iter_chunks()


class ImmitationDataSet_hdf5_multi(IterableDataset):
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
            cam_framestack = torch.vstack(tuple(itertools.islice(self.framestack_cam, 0, None, self.frameskip)))
            cam_fixed_framestack = torch.vstack(tuple(itertools.islice(self.framestack_cam, 0, None, self.frameskip)))
            gs_framestack = torch.vstack(tuple(itertools.islice(self.framestack_gel, 0, None, self.frameskip)))
        else:
            cam_framestack = torch.vstack(([self.framestack_cam[-1]] * self.num_stack))
            cam_fixed_framestack = torch.vstack(([self.framestack_cam[-1]] * self.num_stack))
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
    parser.add_argument("--log_file", default="train.csv")
    parser.add_argument("--num_stack", default=5)
    parser.add_argument("--frameskip", default=2)
    parser.add_argument("--data_folder", default="data/test_recordings_0208_repeat")
    args = parser.parse_args()
    # dataset = TripletDataset(args.log_file)
    # cam_pos, gs_pos, log_spec, cam_neg = dataset[53]

    # dataset = ImmitationDataSet(args.log_file)
    # loader = DataLoader(dataset, 4, num_workers=1)
    # for _ in loader:
    #     pass
    # dataset = FuturePredDataset("train.csv", 10)
    # for cam_frames, log_specs, gs_frames, actions in dataset:
    #     pass

    dataset = ImmitationDataSet_hdf5("train.csv")
    # print("dataset", dataset.len)
    cnt = 0
    for cam_frame, _, _ in dataset:
        cnt+=1
        if cnt >=5:
            break