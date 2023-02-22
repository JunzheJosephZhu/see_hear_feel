import os
import torch
import torchvision.transforms as T

from src.datasets.base import EpisodeDataset
import numpy as np
from PIL import Image


class ImitationEpisode(EpisodeDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder, train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip
        self.fps = 10
        self.sr = 44100
        self.resolution = (
            self.sr // self.fps
        )  # number of audio samples in one image idx
        # self.audio_len = int(self.resolution * (max(self.max_len + 1, 10)))
        self.audio_len = self.num_stack * self.frameskip * self.resolution

        # augmentation parameters
        self.EPS = 1e-8
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))
        (
            self.trial,
            self.timestamps,
            self.audio_gripper,
            self.audio_holebase,
            self.num_frames,
        ) = self.get_episode(dataset_idx, ablation=args.ablation)

        # saving the offset for gelsight in order to normalize data
        self.gelsight_offset = (
            torch.as_tensor(
                np.array(Image.open(os.path.join(self.data_folder, "gs_offset.png")))
            )
            .float()
            .permute(2, 0, 1)
            / 255
        )
        self.action_dim = args.action_dim
        self.task = args.task
        self.use_flow = args.use_flow
        self.modalities = args.ablation.split("_")
        self.nocrop = args.nocrop

        if self.train:
            self.transform_cam = [
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
            ]
            self.transform_gel = [
                T.Resize((self.resized_height_t, self.resized_width_t)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            self.transform_cam = T.Compose(self.transform_cam)
            self.transform_gel = T.Compose(self.transform_gel)

        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v)),
                    T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                ]
            )
            self.transform_gel = T.Compose(
                [
                    T.Resize((self.resized_height_t, self.resized_width_t)),
                    T.CenterCrop((self._crop_height_t, self._crop_width_t)),
                ]
            )

    def __len__(self):
        return self.num_frames

    def get_demo(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        if self.task == "pouring":
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            dy_space = {-0.0012: 0, 0: 1, 0.004: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            y_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            z_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            keyboard = (
                x_space[keyboard[0]] * 9
                + y_space[keyboard[1]] * 3
                + z_space[keyboard[2]]
            )
        return keyboard

    def __getitem__(self, idx):
        start = idx - self.max_len
        # compute which frames to use
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = -1
        # images
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        cam_fixed_framestack = 0
        tactile_framestack = 0

        # load first frame for better alignment
        if self.use_flow:
            offset = torch.from_numpy(
                torch.load(
                    os.path.join(self.trial, "left_gelsight_flow", str(0) + ".pt")
                )
            ).type(torch.FloatTensor)
        else:
            offset = self.load_image(self.trial, "left_gelsight_frame", 0)

        # process different streams of data
        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.trial, "cam_gripper_color", timestep)
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )
        if "vf" in self.modalities:
            cam_fixed_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.trial, "cam_fixed_color", timestep)
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )
        if "t" in self.modalities:
            if self.use_flow:
                tactile_framestack = torch.stack(
                    [
                        torch.from_numpy(
                            torch.load(
                                os.path.join(
                                    self.trial,
                                    "left_gelsight_flow",
                                    str(timestep) + ".pt",
                                )
                            )
                        ).type(torch.FloatTensor)
                        - offset
                        for timestep in frame_idx
                    ],
                    dim=0,
                )
            else:
                tactile_framestack = torch.stack(
                    [
                        (
                            self.transform_gel(
                                self.load_image(
                                    self.trial, "left_gelsight_frame", timestep
                                )
                                - offset
                                + 0.5
                            ).clamp(0, 1)
                        )
                        for timestep in frame_idx
                    ],
                    dim=0,
                )
                for i, timestep in enumerate(frame_idx):
                    if timestep < 0:
                        tactile_framestack[i] = torch.zeros_like(tactile_framestack[i])

        # random cropping
        if self.train:
            img = self.transform_cam(
                self.load_image(self.trial, "cam_fixed_color", idx)
            )
            if not self.nocrop:
                i_v, j_v, h_v, w_v = T.RandomCrop.get_params(
                    img, output_size=(self._crop_height_v, self._crop_width_v)
                )
            else:
                i_v, h_v = (
                    self.resized_height_v - self._crop_height_v
                ) // 2, self._crop_height_v
                j_v, w_v = (
                    self.resized_width_v - self._crop_width_v
                ) // 2, self._crop_width_v

            if "vg" in self.modalities:
                cam_gripper_framestack = cam_gripper_framestack[
                    ..., i_v : i_v + h_v, j_v : j_v + w_v
                ]
            if "vf" in self.modalities:
                cam_fixed_framestack = cam_fixed_framestack[
                    ..., i_v : i_v + h_v, j_v : j_v + w_v
                ]
            if "t" in self.modalities:
                if not self.use_flow:
                    img_t = self.transform_gel(
                        self.load_image(self.trial, "left_gelsight_frame", idx)
                    )
                    if not self.nocrop:
                        i_t, j_t, h_t, w_t = T.RandomCrop.get_params(
                            img_t, output_size=(self._crop_height_t, self._crop_width_t)
                        )
                    else:
                        i_t, h_t = (
                            self.resized_height_t - self._crop_height_t
                        ) // 2, self._crop_height_t
                        j_t, w_t = (
                            self.resized_width_t - self._crop_width_t
                        ) // 2, self._crop_width_t
                    tactile_framestack = tactile_framestack[
                        ..., i_t : i_t + h_t, j_t : j_t + w_t
                    ]

        # load audio
        audio_end = idx * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        if self.audio_gripper is not None:
            audio_clip_g = self.clip_resample(
                self.audio_gripper, audio_start, audio_end
            ).float()
        else:
            audio_clip_g = 0
        if self.audio_holebase is not None:
            audio_clip_h = self.clip_resample(
                self.audio_holebase, audio_start, audio_end
            ).float()
        else:
            audio_clip_h = 0

        # load labels
        keyboard = self.get_demo(idx)
        xyzrpy = torch.Tensor(self.timestamps["pose_history"][idx][:6])
        optical_flow = 0

        return (
            (
                cam_fixed_framestack,
                cam_gripper_framestack,
                tactile_framestack,
                audio_clip_g,
                audio_clip_h,
            ),
            keyboard,
            xyzrpy,
            optical_flow,
            start,
        )


class TransformerEpisode(ImitationEpisode):
    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Do not duplicate first frame for padding, instead return all zeros
        """
        return_null = timestep == -1
        if timestep == -1:
            timestep = 0
        img_path = os.path.join(trial, stream, str(timestep) + ".png")
        image = (
            torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
            / 255
        )
        if return_null:
            image = torch.zeros_like(image)
        return image
