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

class TripletDataset(Dataset):
    def __init__(self, log_file=None, data_folder="data", negative_range=0):
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder

    def __getitem__(self, idx):
        trial = os.path.join(self.data_folder, self.logs.iloc[idx].Time)
        audio1, sr = sf.read(os.path.join(trial, "audio_in1.wav"))
        audio2, sr = sf.read(os.path.join(trial, "audio_in2.wav"))
        resolution = sr // 10 # number of audio samples in each video frame
        # print(audio1.max(), audio1.min(), audio2.max(), audio2.min(), audio1.shape, audio2.shape)
        assert audio1.shape == audio2.shape
        audio = torch.as_tensor(np.stack([audio1, audio2], 0))
        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)
        print(timestamps.keys())
        # read camera frames
        cam_video = cv2.VideoCapture(os.path.join(trial, "cam.avi"))
        success, cam_frame = cam_video.read()
        cam_frames = []
        while success:
            cam_frames.append(cam_frame)
            success, cam_frame = cam_video.read()
        cam_frames = np.stack(cam_frames, 0)
        # read gelsight frames
        gs_video = cv2.VideoCapture(os.path.join(trial, "gs.avi"))
        success, gs_frame = gs_video.read()
        gs_frames = []
        while success:
            gs_frames.append(gs_frame)
            success, gs_frame = gs_video.read()
        gs_frames = torch.as_tensor(np.stack(gs_frames, 0))
        # voice activity detection
        audio_frames = audio.unfold(dimension=-1, size=resolution, step=resolution)
        energy = torch.pow(audio_frames, 2).sum(-1).sum(0)
        plt.plot(energy)
        plt.plot(torch.ones(energy.shape) * 2)
        print(trial)
        plt.show()
        # sample anchor times
        if torch.rand(()) > 0.2: # sample anchor with audio event
            anchor_choices = torch.nonzero(energy > 2)
            anchor = anchor_choices[torch.randint(high=anchor_choices.size(0), size=())]
        else:
            anchor_choices = torch.nonzero(energy < 2)
            anchor = anchor_choices[torch.randint(high=anchor_choices.size(0), size=())]
        # get image and gelsight
        cam_pos = cam_frames[anchor]
        gs_pos = gs_frames[anchor]
        # audio length is 1 second
        audio_start = anchor * resolution - sr // 2
        audio_end = audio_start + sr
        if audio_start < 0:
            audio_start, audio_end = 0, sr
        if audio_end >= audio.size(-1):
            audio_start, audio_end = audio.size(-1) - sr, audio.size(-1)
        audio_pos = audio[:, audio_start: audio_end]
        # sample negative index
        # negative_choices = torch.cat([torch.arange(0, anchor.item()), torch.arange(anchor.item() + 1, energy.size(-1))])
        negative = negative_choices[torch.randint(high=negative_choices.size(0), size=())]
        cam_neg = cam_frames[negative]
        return cam_pos, gs_pos, audio_pos, cam_neg

    def __len__(self):
        return len(self.logs)

if __name__ == "__main__":
    import argparse
    parser = ArgumentParser()
    parser.add_argument('--log_file', default="log_episodes_0111.csv")
    args = parser.parse_args()
    dataset = TripletDataset(args.log_file)
    dataset[11]
    print(len(dataset))