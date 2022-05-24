import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import torchaudio
import soundfile as sf
import time

class BaseDataset(Dataset):
    def __init__(self, log_file, data_folder="data/test_recordings_0214"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self.sr = 44100
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=int(self.sr * 0.025), hop_length=int(self.sr * 0.01), n_mels=64, center=False
        )
        self.streams = ["cam_gripper_color", "cam_fixed_color", "left_gelsight_flow", "left_gelsight_frame"]
        # self.gelsight_offset = torch.as_tensor(np.array(Image.open("gs_offset.png"))).float().permute(2, 0, 1) / 255
        pass


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
            audio_holebase_left = sf.read(os.path.join(trial, 'audio_holebase_left.wav'))[0]
            audio_holebase_right = sf.read(os.path.join(trial, 'audio_holebase_right.wav'))[0]
            audio_gripper = torch.as_tensor(np.stack([audio_gripper_left, audio_gripper_right], 0))
            audio_holebase = torch.as_tensor(np.stack([audio_holebase_left, audio_holebase_right], 0))
        else:
            audio_gripper = None
            audio_holebase = None
        return trial, timestamps, audio_gripper, audio_holebase, len(timestamps["action_history"])


    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        img_path = os.path.join(trial, stream, str(timestep) + ".png")
        image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
        return image

    @staticmethod
    def load_flow(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        img_path = os.path.join(trial, stream, str(timestep) + ".pt")
        image = torch.as_tensor(torch.load(img_path))
        return image

    @staticmethod
    def clip_resample(audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        # audio_clip = torchaudio.functional.resample(audio_clip, 44100, 16000)
        return audio_clip

    def __len__(self):
        return len(self.logs)

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3 # [3, H, W]
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="bilinear").squeeze(0)