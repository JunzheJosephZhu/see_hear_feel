from argparse import ArgumentParser
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
        audio1, sr = sf.read(os.path.join(trial, "audio_in1.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_in2.wav"))
        assert sr == 16000
        resolution = sr // 10  # number of audio samples in each video frame
        # print(audio1.max(), audio1.min(), audio2.max(), audio2.min(), audio1.shape, audio2.shape)
        assert audio1.shape == audio2.shape
        audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()

        # read camera frames
        cam_video = cv2.VideoCapture(os.path.join(trial, "cam.avi"))
        success, cam_frame = cam_video.read()
        cam_frames = []
        while success:
            cam_frames.append(cam_frame)
            success, cam_frame = cam_video.read()
        cam_frames = torch.as_tensor(np.stack(cam_frames, 0))
        # read gelsight frames
        gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))
        success, gs_frame = gs_video.read()
        gs_frames = []
        while success:
            gs_frames.append(gs_frame)
            success, gs_frame = gs_video.read()
        gs_frames = torch.as_tensor(np.stack(gs_frames, 0))
        # see how many frames there are
        assert cam_frames.size(0) == gs_frames.size(0)
        num_frames = cam_frames.size(0)

        # voice activity detection
        audio_frames = audio.unfold(dimension=-1, size=resolution, step=resolution)
        energy = torch.pow(audio_frames, 2).sum(-1).sum(0)
        # plt.plot(energy)
        # plt.plot(torch.ones(energy.shape) * 2)
        # print(trial)
        # plt.show()
        # sample anchor times
        if torch.rand(()) > self.sil_ratio and (energy > 2).any():  # sample anchor with audio event
            anchor_choices = torch.nonzero(energy > 2)
            anchor = anchor_choices[
                torch.randint(high=anchor_choices.size(0), size=())
            ].item()
        else:
            anchor_choices = torch.nonzero(energy < 2)
            anchor = anchor_choices[
                torch.randint(high=anchor_choices.size(0), size=())
            ].item()
        # get image and gelsight
        cam_pos = cam_frames[anchor]
        gs_pos = gs_frames[anchor]
        # audio length is 1 second
        audio_start = anchor * resolution - sr // 2
        audio_end = audio_start + sr
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((2, -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((2, audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_pos = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )

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
        if worker_info is not None:
            assert worker_info.num_workers == 1, "multiworker not supported"
        sr = 16000
        resolution = sr // 10
        success, cam_frame = self.cam_video.read()
        success, gs_frame = self.gs_video.read()
        if not success:
            self.idx += 1
            if self.idx == len(self.logs):
                raise StopIteration
            self.load_episode(self.idx)
            success, cam_frame = self.cam_video.read()
            success, gs_frame = self.gs_video.read()
        assert success
        cam_frame = torch.as_tensor(cam_frame).permute(2, 0, 1) / 255
        gs_frame = torch.as_tensor(gs_frame).permute(2, 0, 1) / 255
        # load audio clip
        # audio length is 1 second
        audio_start = self.timestep * resolution - sr // 2
        audio_end = audio_start + sr
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((2, -audio_start))
            audio_start = 0
        if audio_end >= self.audio.size(-1):
            right_pad = torch.zeros((2, audio_end - self.audio.size(-1)))
            audio_end = self.audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, self.audio[:, audio_start:audio_end], right_pad], dim=1
        )
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
        return cam_frame, gs_frame, log_spec, action

    def load_episode(self, idx):
        # reset timestep
        self.timestep = 0
        # get file older
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        # load audio tracks
        audio1, sr = sf.read(os.path.join(trial, "audio_in1.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_in2.wav"))
        self.audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()
        # load json file
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        # read camera frames
        self.cam_video = cv2.VideoCapture(os.path.join(trial, "cam.avi"))
        # read gelsight frames
        self.gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train.csv")
    args = parser.parse_args()
    dataset = TripletDataset(args.log_file)
    cam_pos, gs_pos, log_spec, cam_neg = dataset[11]

    dataset = ImmitationDataSet(args.log_file)
    loader = DataLoader(dataset, 4, num_workers=1)
    for _ in loader:
        pass