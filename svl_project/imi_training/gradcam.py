
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

import sys
from telnetlib import KERMIT
from tomlkit import key
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch

# from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti, ImitationDatasetLabelCount
from svl_project.datasets.imi_dataset import ImitationDataset, ImitationDatasetLabelCount
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_audio_encoder,make_tactile_flow_encoder
from svl_project.models.imi_models import Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiEngine
from torch.utils.data import DataLoader
from itertools import cycle
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from svl_project.boilerplate import *
import pandas as pd
import numpy as np
import time
from typing import Callable, List, Tuple
import cv2
from PIL import Image

def strip_sd(state_dict, prefix):
    """
    strip prefix from state dictionary
    """
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

class MyGradCAM(GradCAM):
    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = [inp.cuda() for inp in input_tensor] 

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        return 96, 128


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode
        
    train_label_set = torch.utils.data.ConcatDataset([ImitationDatasetLabelCount(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    train_set = torch.utils.data.ConcatDataset([ImitationDataset(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    val_set = torch.utils.data.ConcatDataset([ImitationDataset(args.val_csv, args, i, args.data_folder, False) for i in range(val_num_episode)])

    # create weighted sampler to balance samples
    train_label = []
    for keyboard in train_label_set:
        train_label.append(keyboard)
    class_sample_count = np.zeros(pow(3, args.action_dim))
    for t in np.unique(train_label):
        class_sample_count[t] = len(np.where(train_label == t)[0])
    weight = 1. / (class_sample_count + 1e-5)
    samples_weight = np.array([weight[t] for t in train_label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    # train_loader = DataLoader(train_set, args.batch_size, num_workers=8, sampler=sampler, persistent_workers=True, pin_memory=True)
    # val_loader = DataLoader(val_set, 1, num_workers=0, shuffle=False, pin_memory=True)
    
    # v encoder
    v_encoder = make_vision_encoder(args.encoder_dim)
    # t encoder
    if args.use_flow:
        t_encoder = make_tactile_flow_encoder(args.encoder_dim)
    else:
        t_encoder = make_tactile_encoder(args.encoder_dim)
    # a encoder
    a_encoder = make_audio_encoder(args.encoder_dim, args.num_stack, args.norm_audio)
    model = Imitation_Actor_Ablation(v_encoder, t_encoder, a_encoder, args).cuda()

    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.split('.', 1)[-1] : v for k, v in state_dict.items() if k.startswith(prefix)}

    state_dict = strip_sd(torch.load(args.pretrained)['state_dict'], 'actor.')
    model.load_state_dict(state_dict)
    model.eval()
    cam = MyGradCAM(model=model, target_layers=[list(v_encoder.feature_extractor.modules())[-1]], use_cuda=True)

    def show_cam_on_image(img: np.ndarray,
                        mask: np.ndarray,
                        use_rgb: bool = False,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.resize(heatmap, [img.shape[1], img.shape[0]])

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        cam = heatmap + img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    for episode in val_set.datasets:
        episode.to_print=True
    for idx in range(100):
        inputs, demo, xyzrpy_gt, optical_flow = val_set[idx]
        inputs = [tmp.unsqueeze(0) if torch.is_tensor(tmp) else torch.Tensor([tmp]) for tmp in inputs]
        demo = [demo]

        targets = [ClassifierOutputTarget(demo[0])]
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h = inputs
        img = np.array(Image.open("data/data_0607/test_recordings/2022-06-03 22:20:55.269548/cam_gripper_color/0.png"))/255
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imsave(f"gradcam_output/{idx}_img_v.jpg", visualization)

        cam = MyGradCAM(model=model, target_layers=[list(t_encoder.feature_extractor.modules())[-1]], use_cuda=True)
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h = inputs
        img = np.array(Image.open("data/data_0607/test_recordings/2022-06-03 22:20:55.269548/left_gelsight_frame/0.png"))/255
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imsave(f"gradcam_output/{idx}_img_t.jpg", visualization)

        import torchaudio
        sr = 16000
        mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=int(sr * 0.025) + 1, hop_length=int(sr * 0.01), n_mels=256
            )
        cam = MyGradCAM(model=model, target_layers=[list(a_encoder.feature_extractor.modules())[-1]], use_cuda=True)
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h = inputs
        img = torch.chunk(mel(audio_clip_h.float()), 6, -1)[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img /= img.max()
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imsave(f"gradcam_output/{idx}_img_a.jpg", visualization)



if __name__ == "__main__":
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("--batch_size", default=16,type=int)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=60, type=int)
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
    p.add("--use_query", default=False, action="store_true")
    p.add("--use_lstm", default=False, action="store_true")
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/data_0607/test_recordings")
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
    p.add("--minus_first", default=False, action="store_true")
    p.add("--pretrained", required=True, type=str)


    


    args = p.parse_args()
    main(args)

