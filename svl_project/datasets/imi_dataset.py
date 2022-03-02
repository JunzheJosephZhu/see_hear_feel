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
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class ImitationOverfitDataset(BaseDataset):
    def __init__(self, log_file, dataset_idx, data_folder="data/test_recordings"):
        super().__init__(log_file, data_folder)
        self.dataset_idx = dataset_idx
        # method1:get the len of entire dataset and iterate
        # self.ep_idx = 0
        _, _, _, self.num_frames = self.get_episode(self.dataset_idx, load_audio=False)


    def __len__(self):
        return self.num_frames

    def get_episode(self, idx, load_audio=True):
        """
        Return:
            folder for trial
            logs
            audio tracks
            number of frames in episode
        """
        format_time = self.logs.iloc[idx].Time.replace(":", "_")
        # print("override" + '#' * 50)
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)
        if load_audio:
            audio_gripper = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio_hole = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio = torch.as_tensor(np.stack([audio_gripper, audio_hole], 0))
        else:
            audio = None
        return trial, timestamps, audio, len(timestamps["action_history"])

    def __getitem__(self, idx):

        trial, timestamps, _, num_frames = self.get_episode(self.dataset_idx, load_audio=False)
        # if idx == num_frames - 1:
        #     if self.ep_idx == 10:
        #         return StopIteration
        #     self.ep_idx += 1
        #     _,_,_,self.num_frames = self.get_episode(self.ep_idx, load_audio=False)
        timestep = idx #torch.randint(high=num_frames, size=()).item()
        # print(timestep)
        trans = T.Compose([
            T.Resize((160, 120)),
            # T.RandomCrop((self._crop_height, self._crop_width)),
            # T.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.1),
        ])
        trans2 = T.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.1)
        cam_gripper_color = trans(self.load_image(trial, "cam_gripper_color", timestep))
        cam_fixed_color = trans(self.load_image(trial, "cam_fixed_color", timestep))

        # print("gripper", cam_gripper_color.shape)
        keyboard = timestamps["action_history"][timestep]
        xy_space = {-.003: 0, 0: 1, .003: 2}
        z_space = {-.0015: 0, 0: 1, .0015: 2}
        keyboard = torch.as_tensor([xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]]])
        v_total = torch.stack((cam_gripper_color, cam_fixed_color))
        return v_total, keyboard

class ImitationDatasetSingleCam(BaseDataset):
    def __init__(self, log_file, args, dataset_idx,  device, data_folder="data/test_recordings"):
        super().__init__(log_file, data_folder)
        self.dataset_idx = dataset_idx
        # method1:get the len of entire dataset and iterate
        # self.ep_idx = 0
        self.device = device
        self.resized_height = args.resized_height
        self.resized_width = args.resized_width
        self._crop_height = int(args.resized_height * (1.0 - args.crop_percent))
        self._crop_width = int(args.resized_width * (1.0 - args.crop_percent))
        _, _, _, self.num_frames = self.get_episode(self.dataset_idx, load_audio=False)

    def __len__(self):
        return self.num_frames

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
            audio_gripper = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio_hole = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio = torch.as_tensor(np.stack([audio_gripper, audio_hole], 0))
        else:
            audio = None
        return trial, timestamps, audio, len(timestamps["action_history"])

    def __getitem__(self, idx):

        trial, timestamps, _, num_frames = self.get_episode(self.dataset_idx, load_audio=False)
        # if idx == num_frames - 1:
        #     if self.ep_idx == 10:
        #         return StopIteration
        #     self.ep_idx += 1
        #     _,_,_,self.num_frames = self.get_episode(self.ep_idx, load_audio=False)
        timestep = idx #torch.randint(high=num_frames, size=()).item()
        # print(timestep)
        trans2 = T.Compose([
            # T.Resize((160, 120)),
            T.RandomCrop((self._crop_height, self._crop_width)),
            T.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.1),
        ])
        trans = T.Resize((self.resized_height, self.resized_width))
        # cam_gripper_color = trans(self.load_image(trial, "cam_gripper_color", timestep))
        cam_fixed_color = trans2(trans(self.load_image(trial, "cam_fixed_color", timestep)))

        # print("gripper", cam_gripper_color.shape)
        keyboard = timestamps["action_history"][timestep]
        xy_space = {-.003: 0, 0: 1, .003: 2}
        z_space = {-.0015: 0, 0: 1, .0015: 2}
        keyboard = torch.as_tensor([xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]]])
        # v_total = torch.stack((cam_gripper_color, cam_fixed_color))
        cam_fixed_color = torch.unsqueeze(cam_fixed_color, 0)
        return cam_fixed_color, keyboard

    
class ImitationDatasetFramestack(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, device, data_folder="data/test_recordings"):
        super().__init__(log_file, data_folder)
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.resized_height = args.resized_height
        self.resized_width = args.resized_width
        self._crop_height = int(args.resized_height * (1.0 - args.crop_percent))
        self._crop_width = int(args.resized_width * (1.0 - args.crop_percent))
        self.max_len = (self.num_stack - 1) * self.frameskip + 1
        self.trial, self.timestamps, _, self.num_frames = self.get_episode(dataset_idx, load_audio=False)
        self.device = device
    
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
            audio_gripper = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio_hole = sf.read(os.path.join(trial, 'audio_gripper.wav'))[0]
            audio = torch.as_tensor(np.stack([audio_gripper, audio_hole], 0))
        else:
            audio = None
        return trial, timestamps, audio, len(timestamps["action_history"])

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        end = idx #torch.randint(high=num_frames, size=()).item()
        start = end - self.max_len
        if start < 0:
            cam_idx = [end] * self.num_stack
        else:
            cam_idx = list(np.arange(start, end, self.frameskip))

        transform = T.Compose([
            T.Resize((self.resized_height, self.resized_width)),
            # T.RandomCrop((self._crop_height, self._crop_width)),
            # T.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.1),
        ])
        # transform2 = T.Compose([
        #     # T.Resize((self.resized_height, self.resized_width)),
        #     # T.RandomCrop((self._crop_height, self._crop_width)),
        #     T.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.1),
        # ])
        img = transform(self.load_image(self.trial, "cam_gripper_color", end))
        i, j, h, w = T.RandomCrop.get_params(img, output_size=(self._crop_height, self._crop_width))

        # transform = T.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.05)
        # t_start = time.time()
        cam_gripper_framestack = torch.stack(
            [T.functional.crop(transform(self.load_image(self.trial, "cam_gripper_color", timestep)).to(self.device), i, j, h, w) for timestep in cam_idx], dim=0)
            # [T.functional.crop(self.load_image(self.trial, "cam_gripper_color", timestep), i, j, h, w) for timestep in cam_idx], dim=0)

        cam_fixed_framestack = torch.stack(
            [T.functional.crop(transform(self.load_image(self.trial, "cam_fixed_color", timestep).to(self.device)), i, j, h ,w) for timestep in cam_idx],dim=0)
            # [T.functional.crop(self.load_image(self.trial, "cam_fixed_color", timestep), i, j, h, w) for timestep in cam_idx], dim=0)
        # print(time.time() - t_start)

        keyboard = self.timestamps["action_history"][end]
        xy_space = {-.003: 0, 0: 1, .003: 2}
        z_space = {-.0015: 0, 0: 1, .0015: 2}
        keyboard = torch.as_tensor([xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]]])
        v_total = torch.cat((cam_gripper_framestack, cam_fixed_framestack), dim=0)
        return v_total, keyboard
    
    # def load_image(self, trial, stream, timestep):
    #     """
    #     Args:
    #         trial: the folder of the current episode
    #         stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
    #             for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
    #         timestep: the timestep of frame you want to extract
    #     """
        
    #     img_path = os.path.join(trial, stream, str(timestep) + ".png")
    #     if img_path in self.img_set:
    #         return self.img_set[img_path]
    #     else:
    #         image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
    #         self.img_set[img_path] = image
    #         return image

class ImitationDatasetFramestackMulti(BaseDataset):
    def __init__(self, log_file, args, data_folder="data/test_recordings_0214"):
        super().__init__(log_file, data_folder)
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip + 1
        self.fps = 10
        self.sr = int(16000 * (self.fps / max(self.max_len, 10)))
        self.resolution = self.sr // 10
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=int(self.sr * 0.025), hop_length=int(self.sr * 0.01), n_mels=64
        )
        self.EPS = 1e-8

    def __getitem__(self, idx):
        trial, timestamps, audio, num_frames = self.get_episode(idx, load_audio=True)

        # load camera frames
        end = torch.randint(high=num_frames, size=()).item()
        start = end - self.max_len
        if start < 0:
            cam_idx = [end] * self.num_stack
        else:
            cam_idx = list(np.arange(start, end, self.frameskip))

        cam_gripper_framestack = torch.cat(
            [self.load_image(trial, "cam_gripper_color", timestep) for timestep in cam_idx], dim=0)

        cam_fixed_framestack = torch.cat(
            [self.load_image(trial, "cam_fixed_color", timestep) for timestep in cam_idx], dim=0)

        # load audio
        audio_start = end * self.resolution - self.sr // 2 # why self.sr // 2, and start + sr
        audio_end = audio_start + self.sr
        audio_clip = self.clip_audio(audio, audio_start, audio_end)
        spec = self.mel(audio_clip)
        log_spec = torch.log(spec + EPS)

        keyboard = timestamps["action_history"][end]
        xy_space = {-.003: 0, 0: 1, .003: 2}
        z_space = {-.0015: 0, 0: 1, .0015: 2}
        keyboard = torch.as_tensor([xy_space[keyboard[0]], xy_space[keyboard[1]], z_space[keyboard[2]]])

        return cam_gripper_framestack, cam_fixed_framestack, log_spec, keyboard


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
        audio_clip = self.clip_audio(self.audio, audio_start, audio_end)
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
    parser.add_argument("--num_stack", default=5, type=int)
    parser.add_argument("--frameskip", default=2, type=int)
    parser.add_argument("--data_folder", default="data/test_recordings_0220_toy")
    args = parser.parse_args()

    dataset = ImitationOverfitDataset("train.csv")
    # print("dataset", dataset.len)
    cnt = 0
    zero_cnt = 0
    t_l = []
    num_frame = 0
    for _ in range(11800):
        index = torch.randint(high=10,size=()).item()
        _, _, _, idx, t, num = dataset.__getitem__(index)
        if idx == 0:
            num_frame = num
            zero_cnt += 1
            t_l.append(t)
        cnt += 1
    mydic = {i:t_l.count(i) for i in t_l}
    print(zero_cnt)
    print(num_frame)
    print(len(mydic))

    print(mydic)