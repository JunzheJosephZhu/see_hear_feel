
from svl_project.datasets.imi_dataset import ImitationDataset
import numpy as np
import os
import torch
from PIL import Image
class TransformerDataset(ImitationDataset):
    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        return_null = timestep == -1
        if timestep == -1:
            timestep = 0
        img_path = os.path.join(trial, stream, str(timestep) + ".png")
        image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
        if return_null:
            image = torch.zeros_like(image)
        return image

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
    dataset = TransformerDataset(args.train_csv, args, 0, args.data_folder)
    for (cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h), keyboard, xyzrpy, optical_flow, start in dataset:
        # print(cam_fixed_framestack.shape)
        print(start)