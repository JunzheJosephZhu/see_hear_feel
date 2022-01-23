from argparse import ArgumentParser
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import soundfile as sf
import numpy as np
import json
import cv2
import time
import matplotlib.pyplot as plt
import torch
import torchaudio

EPS = 1e-8


class TripletDataset(Dataset):
    def __init__(self, log_file=None, data_folder="data"):
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )

    def __getitem__(self, idx):
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        audio1, sr = sf.read(os.path.join(trial, "audio_in1.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_in2.wav"))
        assert sr == 16000
        resolution = sr // 10  # number of audio samples in each video frame
        # print(audio1.max(), audio1.min(), audio2.max(), audio2.min(), audio1.shape, audio2.shape)
        assert audio1.shape == audio2.shape
        audio = torch.as_tensor(np.stack([audio1, audio2], 0)).float()

        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)
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
        if torch.rand(()) > 0.2 and (energy > 2).any():  # sample anchor with audio event
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


if __name__ == "__main__":
    import argparse

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train.csv")
    args = parser.parse_args()
    dataset = TripletDataset(args.log_file)
    cam_pos, gs_pos, log_spec, cam_neg = dataset[11]
    print(gs_pos.dtype)