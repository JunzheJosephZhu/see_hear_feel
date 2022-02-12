import itertools
from argparse import ArgumentParser
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import os
import soundfile as sf
import torch
import h5py

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

def convert2hdf5(filename):
    # read camera frames
    cam_video = cv2.VideoCapture(os.path.join(filename, "cam_gripper.avi"))
    f1 = h5py.File("test_data.hdf5", "w")
    cnt = 1
    _, cam_frame = cam_video.read()
    cam_frame = torch.tensor(cam_frame)
    cam_tensors = torch.unsqueeze(cam_frame, 0)
    while True:
        success, cam_frame = cam_video.read()
        if not success:
            break
        cam_frame = torch.tensor(cam_frame)
        cam_frame = torch.unsqueeze(cam_frame, 0)
        cam_tensors = torch.cat((cam_tensors, cam_frame), dim=0)
        cnt += 1

    dset1 = f1.create_dataset("dataset", (cnt, 3, 480, 640), dtype='f', data=cam_tensors)


class ImmitationDataSet_Tuning(IterableDataset):
    def __init__(self):
        super().__init__()
        self.idx = 0
        self.open_hdf5()
        self.maxlen = 99


    def __iter__(self):
        return self

    def __next__(self):
        assert self.idx <= 5
        cam_frame = self.img_hdf5['cam_gripper_color'][next(self.dataset)]
        plt.imshow(cam_frame)
        plt.show()
        cam_frame = torch.as_tensor(cam_frame).permute(2, 0, 1) / 255

        self.idx += 1
        return cam_frame

    def open_hdf5(self):
        self.img_hdf5 = h5py.File('data/test_recordings_0208_repeat/2022-02-08 19_36_51.349856/data.hdf5', 'r')
        self.dataset = self.img_hdf5['cam_gripper_color'].iter_chunks()

    def __len__(self):
        return self.maxlen


if __name__ == "__main__":
    # convert2hdf5("data/test_recordings_0208_repeat/2022-02-08 19_36_51.349856")
    dataset = ImmitationDataSet_Tuning()
    # print("dataset", dataset.len)
    for cam_frame in dataset:
        print(cam_frame)