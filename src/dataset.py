import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from argparse import ArgumentParser
from re import L
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import os
import soundfile as sf
import numpy as np
import json
import cv2
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
import torchaudio
import math

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


class TripletDataset(Dataset):
    def __init__(self, log_file, sil_ratio=0.2, data_folder="data"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )
        self.sil_ratio = sil_ratio

    def __getitem__(self, idx):
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        audio1, sr = sf.read(os.path.join(trial, "audio_holebase.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_gripper.wav"))
        assert sr == 16000
        resolution = sr // 10  # number of audio samples in each video frame
        # print(audio1.max(), audio1.min(), audio2.max(), audio2.min(), audio1.shape, audio2.shape)
        assert audio1.shape == audio2.shape
        audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()

        # read camera frames
        cam_video = cv2.VideoCapture(os.path.join(trial, "cam_gripper.avi"))
        success, cam_frame = cam_video.read()
        cam_frames = []
        while success:
            cam_frames.append(cam_frame)
            success, cam_frame = cam_video.read()
        cam_frames = torch.as_tensor(np.stack(cam_frames, 0))
        # print("cam_frames shape: {}".format(cam_frames.shape))

        # read gelsight frames
        gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))
        success, gs_frame = gs_video.read()
        gs_frames = []
        while success:
            gs_frames.append(gs_frame)
            success, gs_frame = gs_video.read()
        gs_frames = torch.as_tensor(np.stack(gs_frames, 0))
        # print("gs_frames shape: {}".format(gs_frames.shape))
        # see how many frames there are
        assert cam_frames.size(0) == gs_frames.size(0)
        num_frames = cam_frames.size(0)

        # voice activity detection
        audio_frames = audio.unfold(dimension=-1, size=resolution, step=resolution)
        energy = torch.pow(audio_frames, 2).sum(-1).sum(0) # user the gripper piezo
        # plt.plot(energy)
        # plt.plot(torch.ones(energy.shape) * 2.5)
        # print(trial)
        # plt.show()
        # sample anchor times
        if torch.rand(()) > self.sil_ratio and (energy > 2.5).any():  # sample anchor with audio event
            anchor_choices = torch.nonzero(energy > 2.5)
            anchor = anchor_choices[
                torch.randint(high=anchor_choices.size(0), size=())
            ].item()
        else:
            anchor_choices = torch.nonzero(energy < 2.5)
            anchor = anchor_choices[
                torch.randint(high=anchor_choices.size(0), size=())
            ].item()
        # get image and gelsight
        cam_pos = cam_frames[anchor]
        gs_pos = gs_frames[anchor]
        # audio length is 1 second
        audio_start = anchor * resolution - sr // 2
        audio_end = audio_start + sr
        audio_pos = clip_audio(audio, audio_start, audio_end)

        assert audio_pos.size(1) == sr
        spec = self.mel(audio_pos)
        log_spec = torch.log(spec + EPS)

        # sample negative index
        upper_bound = anchor - 5
        lower_bound = anchor + 5
        negative_range = torch.Tensor([]).int()
        if upper_bound > 0:
            negative_range = torch.cat([negative_range, torch.arange(0, upper_bound)])
        if lower_bound < num_frames:
            negative_range = torch.cat(
                [negative_range, torch.arange(lower_bound, num_frames)]
            )
        negative = negative_range[torch.randint(high=negative_range.size(0), size=())]
        cam_neg = cam_frames[negative]

        return (
            cam_pos.permute(2, 0, 1) / 255,
            gs_pos.permute(2, 0, 1) / 255,
            log_spec,
            cam_neg.permute(2, 0, 1) / 255,
        )

    def __len__(self):
        return len(self.logs)

class ImmitationDataSet(IterableDataset):
    def __init__(self, log_file=None, data_folder="data"):
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )
        self.iter_end = len(self.logs)
        self.idx = 0
        self.load_episode(self.idx)

    def __iter__(self):
        return self

    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     assert worker_info.num_workers == 1, "multiworker not supported"
        sr = 16000
        resolution = sr // 10
        success, cam_frame = self.cam_video.read()
        success, cam_fixed_frame = self.cam_fixed_video.read()
        success, gs_frame = self.gs_video.read()
        if not success:
            self.cam_video.release()
            self.cam_fixed_video.release()
            self.gs_video.release()
            self.idx += 1
            if self.idx == len(self.logs):
                self.idx = 0
                self.load_episode(0)
                raise StopIteration
            self.load_episode(self.idx)
            success, cam_frame = self.cam_video.read()
            assert success
            success, cam_fixed_frame = self.cam_fixed_video.read()
            assert success
            success, gs_frame = self.gs_video.read()
            assert success
        assert success
        cam_frame = torch.as_tensor(cam_frame).permute(2, 0, 1) / 255
        cam_fixed_frame = torch.as_tensor(cam_fixed_frame).permute(2, 0, 1) / 255
        gs_frame = torch.as_tensor(gs_frame).permute(2, 0, 1) / 255
        # load audio clip
        # audio length is 1 second
        audio_start = self.timestep * resolution - sr // 2
        audio_end = audio_start + sr
        audio_clip = clip_audio(self.audio, audio_start, audio_end)

        assert audio_clip.size(1) == sr
        spec = self.mel(audio_clip)
        log_spec = torch.log(spec + EPS)
        action_c = self.timestamps["action_history"][self.timestep]
        pose_c = self.timestamps["pose_history"][self.timestep]
        xy_space = {-0.006: 0, 0: 1, 0.006: 2}
        z_space = {-0.003: 0, 0: 1, 0.003: 2}
        x = xy_space[action_c[0]]
        y = xy_space[action_c[1]]
        z = z_space[action_c[2]]
        action = torch.as_tensor([x, y, z])
        pose = torch.as_tensor([pose_c[0], pose_c[1], pose_c[2]])
        self.timestep += 1
        return cam_frame, cam_fixed_frame, gs_frame, log_spec, action, pose

    def load_episode(self, idx):
        # reset timestep
        self.timestep = 0
        # get file older
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        # load audio tracks
        audio1, sr = sf.read(os.path.join(trial, "audio_holebase.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_gripper.wav"))
        self.audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()
        # load json file
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        # read camera frames
        self.cam_video = cv2.VideoCapture(os.path.join(trial, "cam_gripper.avi"))
        self.cam_fixed_video = cv2.VideoCapture(os.path.join(trial, "cam_fixed.avi"))
        # read gelsight frames
        self.gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))


class FuturePredDataset(Dataset):
    def __init__(self, log_file, max_len, data_folder="data"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )
        self.max_len = max_len

    def __getitem__(self, idx):
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        audio1, sr = sf.read(os.path.join(trial, "audio_holebase.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_gripper.wav"))
        assert sr == 16000
        resolution = sr // 10  # number of audio samples in each video frame
        # print(audio1.max(), audio1.min(), audio2.max(), audio2.min(), audio1.shape, audio2.shape)
        assert audio1.shape == audio2.shape
        audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()

        # read camera frames
        cam_video = cv2.VideoCapture(os.path.join(trial, "cam_gripper.avi"))
        success, cam_frame = cam_video.read()
        cam_frames = []
        while success:
            cam_frames.append(cam_frame)
            success, cam_frame = cam_video.read()
        cam_frames = torch.as_tensor(np.stack(cam_frames, 0)).permute(0, 3, 1, 2) / 255

        # print("cam_frames shape: {}".format(cam_frames.shape))

        # read gelsight frames
        gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))
        success, gs_frame = gs_video.read()
        gs_frames = []
        while success:
            gs_frames.append(gs_frame)
            success, gs_frame = gs_video.read()
        gs_frames = torch.as_tensor(np.stack(gs_frames, 0)).permute(0, 3, 1, 2) / 255

        # print("gs_frames shape: {}".format(gs_frames.shape))
        # see how many frames there are
        assert cam_frames.size(0) == gs_frames.size(0)
        num_frames = cam_frames.size(0)

        # clip number of frames used for training
        if num_frames > self.max_len and self.max_len > 0:
            start = torch.randint(high=num_frames - self.max_len + 1, size=())
            end = start + self.max_len
        else:
            start = 0
            end = num_frames

        cam_frames = cam_frames[start:end]
        gs_frames = gs_frames[start:end]

        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)

        log_specs = []
        actions = []

        for timestep in range(start, end):
            # get log audio spectrum
            audio_start = timestep * resolution - sr // 2
            audio_end = audio_start + sr
            audio_clip = clip_audio(audio, audio_start, audio_end)
            assert audio_clip.size(1) == sr
            spec = self.mel(audio_clip)
            log_spec = torch.log(spec + EPS)
            # get action
            action_c = timestamps["action_history"][timestep]
            xy_space = {-0.006: 0, 0: 1, 0.006: 2}
            z_space = {-0.003: 0, 0: 1, 0.003: 2}
            x = xy_space[action_c[0]]
            y = xy_space[action_c[1]]
            z = z_space[action_c[2]]
            action = torch.as_tensor([x, y, z])

            log_specs.append(log_spec)
            actions.append(action)
        log_specs = torch.stack(log_specs, dim=0)
        actions = torch.stack(actions, dim=0)

        return cam_frames, log_specs, gs_frames, actions

    def __len__(self):
        return len(self.logs)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train.csv")
    args = parser.parse_args()

    dataset = FuturePredDataset("train.csv", 10)
    for cam_frames, log_specs, gs_frames, actions in dataset:
        pass
