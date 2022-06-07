from argparse import ArgumentParser
import os
import torch
import torchvision.transforms as T

import torchaudio
from svl_project.datasets.base import BaseDataset
import numpy as np
import random
from PIL import Image, ImageEnhance
import time
import cv2
import matplotlib.pyplot as plt

class ImitationDatasetLabelCount(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder=None):
        super().__init__(log_file, data_folder)
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(
            dataset_idx, load_audio=False, ablation=args.ablation)
        self.task = args.task

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        if self.task == "pouring":
            x_space = {-.0003: 0, 0: 1, 0.0003: 2}
            dy_space = {-.0012: 0, 0: 1, .004: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-.0003: 0, 0: 1, .0003: 2}
            y_space = {-.0003: 0, 0: 1, .0003: 2}
            z_space = {-.001: 0, 0: 1, .001: 2}
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
        self.trial, self.timestamps, self.audio_gripper, self.audio_holebase, self.num_frames = self.get_episode(dataset_idx, load_audio=True, ablation=args.ablation)
        
        # saving the offset
        self.gelsight_offset = torch.as_tensor(
            np.array(Image.open(os.path.join(self.data_folder, 'gs_offset.png')))).float().permute(2, 0,
                                                                                                   1) / 255
        self.task = args.task
        self.minus_first = args.minus_first
        self.use_flow = args.use_flow

        self.modalities = args.ablation.split('_')

        if self.train:
            self.start_frame = 0
            self.transform_cam = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ])
            self.transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ])
            
        else:
            self.start_frame = 0 #self.num_frames - 200
            self.transform_cam = T.Compose([
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.CenterCrop((self._crop_height_v, self._crop_width_v))
            ])
            self.transform_gel = T.Compose([
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.CenterCrop((self._crop_height_t, self._crop_width_t))
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
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        cam_fixed_framestack = 0
        tactile_framestack = 0
        # load first frame for better alignment
        if self.minus_first:
            if self.use_flow:
                offset = torch.from_numpy(
                        torch.load(os.path.join(self.trial, "left_gelsight_flow", str(0) + ".pt"))).type(
                        torch.FloatTensor)
            else:
                offset = self.load_image(self.trial, "left_gelsight_frame", 0)
        else:
            if self.use_flow:
                offset = torch.from_numpy(
                        torch.load(os.path.join(self.data_folder, "flow_offset.pt"))).type(
                        torch.FloatTensor)
            else:
                offset = self.gelsight_offset

        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                    [self.transform_cam(self.load_image(self.trial, "cam_gripper_color", timestep))
                        for timestep in frame_idx], dim=0)
        if "vf" in self.modalities:
            cam_fixed_framestack = torch.stack(
                    [self.transform_cam(self.load_image(self.trial, "cam_fixed_color", timestep))
                    for timestep in frame_idx], dim=0)
        if "t" in self.modalities:       
            if self.use_flow:
                tactile_framestack = torch.stack(
                    [torch.from_numpy(
                        torch.load(os.path.join(self.trial, "left_gelsight_flow", str(timestep) + ".pt"))).type(
                        torch.FloatTensor) - offset
                    for timestep in frame_idx], dim=0)
            else:     
                tactile_framestack = torch.stack(
                    [(self.transform_gel(
                        self.load_image(self.trial, "left_gelsight_frame", timestep) - offset
                     + 0.5).clamp(0, 1)) for
                    timestep in frame_idx], dim=0)
                
                # img = (self.transform_gel(
                #         self.load_image(self.trial, "left_gelsight_frame", end) - offset
                #      + 0.5).clamp(0, 1)).numpy().transpose(1, 2, 0)
                # img = self.load_image(self.trial, "cam_gripper_color", end).numpy().transpose(1, 2, 0)         
                # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # # cv2.imshow('asda', img)
                # # cv2.waitKey(10000)
                # plt.imshow(img)
                # plt.show()
        img = self.load_image(self.trial, "left_gelsight_frame", end).numpy().transpose(1, 2, 0)         
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('asda', img)
        # cv2.waitKey(10000)
        plt.imshow(img)
        plt.show()
        # random cropping
        if self.train:
            img = self.transform_cam(self.load_image(self.trial, "cam_fixed_color", end))
            i_v, j_v, h_v, w_v = T.RandomCrop.get_params(img, output_size=(self._crop_height_v, self._crop_width_v))
            if "vg"in self.modalities:
                cam_gripper_framestack = cam_gripper_framestack[..., i_v: i_v + h_v, j_v: j_v+w_v]
            if "vf"in self.modalities:
                cam_fixed_framestack = cam_fixed_framestack[..., i_v: i_v + h_v, j_v: j_v+w_v]
            if "t" in self.modalities:
                if not self.use_flow:
                    img_t = self.transform_gel(self.load_image(self.trial, "left_gelsight_frame", end))
                    i_t, j_t, h_t, w_t = T.RandomCrop.get_params(img_t, output_size=(self._crop_height_t, self._crop_width_t))
                    tactile_framestack = tactile_framestack[..., i_t: i_t + h_t, j_t: j_t+w_t]
                  
        # load audio
        audio_end = end * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
    
        # to speed up data loading, do not resample audio if not using
        audio_clip_g = 0
        audio_clip_h = 0
        if "ag" in self.modalities:
            audio_clip_g = self.clip_resample(self.audio_gripper, audio_start, audio_end)
        # we are only using left holebase, so only return this channel, audio encoder has been changed from 4 to 3
        if "ah" in self.modalities:
            # spoiled code: now using left holebase mic
            audio_clip_h = self.clip_resample(self.audio_holebase, audio_start, audio_end)
        
        # save example
        save_idx = 100
        if not self.train:
            if idx == save_idx:
                if not os.path.exists("figures"):
                    os.mkdir("figures")
                # visual
                cam_fixed_tmp = torch.stack(
                    [self.load_image(self.trial, "cam_fixed_color", timestep)
                    for timestep in frame_idx], dim=0)
                for id, img in enumerate(cam_fixed_tmp):
                    array = img.permute(1, 2, 0).detach().cpu().numpy()
                    plt.imsave(f"figures/v{id}.jpg", array)
                # tactile
                tactile_tmp = torch.stack(
                    [(self.load_image(self.trial, "left_gelsight_frame", timestep) - offset
                     + 0.5).clamp(0, 1) for
                    timestep in frame_idx], dim=0)
                for id, img in enumerate(tactile_tmp):
                    array = img.permute(1, 2, 0).detach().cpu().numpy()
                    plt.imsave(f"figures/t{id}.jpg", array)
                sr = 16000
                self.n_mels = 301
                self.mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_fft=int(sr * 0.025) + 1, hop_length=int(sr * 0.002), n_mels=self.n_mels
                )                 
                spec = self.mel(audio_clip_h.float())
                EPS = 1e-8
                log_spec = torch.log(spec + EPS)
                # audio
                audio_frames = log_spec.chunk(len(frame_idx), -1)
                for id, frame in enumerate(audio_frames):
                    array = frame.squeeze(0).detach().cpu().numpy()
                    plt.imsave(f"figures/a{id}.jpg", array)

        # load labels
        keyboard = self.timestamps["action_history"][end]
        if self.task == "pouring":
            x_space = {-.0003: 0, 0: 1, 0.0003: 2}
            dy_space = {-.0012: 0, 0: 1, .004: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-.0003: 0, 0: 1, .0003: 2}
            y_space = {-.0003: 0, 0: 1, .0003: 2}
            z_space = {-.001: 0, 0: 1, .001: 2}
            keyboard = x_space[keyboard[0]] * 9 + y_space[keyboard[1]] * 3 + z_space[keyboard[2]]
        # 6 D pose
        xyzrpy = np.asarray(self.timestamps["pose_history"][end])[:-1].astype(np.float32)
        optical_flow = 0
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