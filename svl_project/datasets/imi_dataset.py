from argparse import ArgumentParser
import os
import torch
import torchvision.transforms as T

import torchaudio
from svl_project.datasets.base import BaseDataset
import numpy as np
import random
from PIL import Image, ImageEnhance

class ImitationDatasetLabelCount(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder=None):
        super().__init__(log_file, data_folder)
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(
            dataset_idx, load_audio=False)
        self.task = args.task

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        if self.task == "pouring":
            x_space = {-.0008: 0, 0: 1, .0008: 2}
            dy_space = {-.006: 0, 0: 1, .006: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-.0004: 0, 0: 1, .0004: 2}
            y_space = {-.0004: 0, 0: 1, .0004: 2}
            z_space = {-.0009: 0, 0: 1, .0009: 2}
            keyboard = x_space[keyboard[0]] * 9 + y_space[keyboard[1]] * 3 + z_space[keyboard[2]]
        return keyboard


class ImitationDataset(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder="data/test_recordings_0214", train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip
        self.fps = 10
        self.sr = 44100
        self.resolution = self.sr // self.fps  # number of audio samples in one image idx
        self.audio_len = int(self.resolution * (max(self.max_len + 1, 10)))

        self.EPS = 1e-8
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(dataset_idx, load_audio=True)
        
        # saving the offset
        self.gelsight_offset = torch.as_tensor(
            np.array(Image.open(os.path.join(self.data_folder, 'gs_offset.png')))).float().permute(2, 0,
                                                                                                   1) / 255
        self.action_dim = args.action_dim
        self.task = args.task
        
        if self.train:
            self.start_frame = 0
            self.transform_cam = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02, hue=0.2),
            ])
            self.transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
            ])
            
        else:
            self.start_frame = 240
            self.transform_cam = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.CenterCrop((self.resized_height_v, self.resized_width_v))
            ])
            self.transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
            ])


    def __len__(self):
        return self.num_frames - self.start_frame

    def __getitem__(self, idx):
        idx += self.start_frame
        end = idx
        start = end - self.max_len
        # compute which frames to use
        frame_idx = np.arange(start, end + 1, self.frameskip)
        frame_idx[frame_idx < 0] = 0

        # images
        cam_gripper_framestack = torch.stack(
                [self.transform_cam(self.load_image(self.trial, "cam_gripper_color", timestep))
                    for timestep in frame_idx], dim=0)
        cam_fixed_framestack = torch.stack(
                [self.transform_cam(self.load_image(self.trial, "cam_fixed_color", timestep))
                 for timestep in frame_idx], dim=0)
        tactile_framestack = torch.stack(
            [(self.transform_gel(
                self.load_image(self.trial, "left_gelsight_frame", timestep) - self.gelsight_offset
            ) + 0.5).clamp(0, 1) for
            timestep in frame_idx], dim=0)

        # random cropping
        if self.train:
            img = self.transform_cam(self.load_image(self.trial, "cam_fixed_color", end))
            i_v, j_v, h_v, w_v = T.RandomCrop.get_params(img, output_size=(self._crop_height_v, self._crop_width_v))
            cam_gripper_framestack = cam_gripper_framestack[..., i_v: i_v + h_v, j_v: j_v+w_v]

        # load audio
        audio_end = end * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        audio_clip_g = self.clip_resample(self.audio_gripper, audio_start, audio_end)
        audio_clip_h = self.clip_resample(self.audio_holebase, audio_start, audio_end)

        # load labels
        keyboard = self.timestamps["action_history"][end]
        if self.task == "pouring":
            x_space = {-.0008: 0, 0: 1, .0008: 2}
            dy_space = {-.006: 0, 0: 1, .006: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-.0004: 0, 0: 1, .0004: 2}
            y_space = {-.0004: 0, 0: 1, .0004: 2}
            z_space = {-.0009: 0, 0: 1, .0009: 2}
            keyboard = x_space[keyboard[0]] * 9 + y_space[keyboard[1]] * 3 + z_space[keyboard[2]]
        xyzrpy = self.timestamps["pose_history"][end]
        optical_flow = None

        return (cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h), keyboard, xyzrpy, optical_flow

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--log_file", default="train.csv")
    parser.add_argument("--num_stack", default=5, type=int)
    parser.add_argument("--frameskip", default=2, type=int)
    parser.add_argument("--data_folder", default="data/test_recordings_0220_toy")
    args = parser.parse_args()

    dataset = ImitationDataset("train.csv")
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