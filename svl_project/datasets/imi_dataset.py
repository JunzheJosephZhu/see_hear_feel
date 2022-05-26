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
        # self.audio_len = int(self.resolution * (max(self.max_len + 1, 10)))
        self.audio_len = self.num_stack * self.frameskip * self.resolution

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
        frame_idx[frame_idx < 0] = -1
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
        audio_clip_g = self.clip_resample(self.audio_gripper, audio_start, audio_end).float()
        audio_clip_h = self.clip_resample(self.audio_holebase, audio_start, audio_end).float()

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
        xyzrpy = torch.Tensor(self.timestamps["pose_history"][end][:6])
        optical_flow = 0

        return (cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h), keyboard, xyzrpy, optical_flow, start

if __name__ == "__main__":
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("--batch_size", default=32)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=65, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--exp_name", required=True, type=str)
    p.add("--encoder_dim", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--use_mha", default=False, action="store_true")
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/data_0502/test_recordings")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--num_heads", required=True, type=int)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    args = p.parse_args()
    dataset = ImitationDataset(args.train_csv, args, 0, args.data_folder)
    for (cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h), keyboard, xyzrpy, optical_flow in dataset:
        # print(cam_fixed_framestack.shape)
        pass