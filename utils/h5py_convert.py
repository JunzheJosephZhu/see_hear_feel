import os
import h5py
import numpy as np
import pandas as pd
from torchvision.utils import save_image
from PIL import Image
import torch
from tqdm import tqdm
from time import time
import soundfile as sf
import cv2

def convert_episode(data_folder, logs, idx):
    print(idx)
    format_time = logs.iloc[idx].Time.replace(":", "_")
    trial = os.path.join(data_folder, format_time)
    all_datasets = h5py.File(os.path.join(trial, "data.hdf5"), 'r')
    streams = ["cam_gripper_color", "cam_fixed_color", "left_gelsight_flow", "left_gelsight_frame"]
    for stream in streams:
        os.makedirs(os.path.join(trial, stream), exist_ok=True)
        frame_chunks = all_datasets[stream].iter_chunks()
        for frame_nb, frame_chunk in enumerate(tqdm(frame_chunks)):
            img = all_datasets[stream][frame_chunk]
            if not stream.endswith("flow"):
                out_file = os.path.join(trial, stream, str(frame_nb) + ".png")
                if not os.path.exists(out_file) or True:
                    cv2.imwrite(out_file, img)
            else:
                out_file = os.path.join(trial, stream, str(frame_nb) + ".pt")
                if not os.path.exists(out_file):
                    torch.save(img, out_file)
    tracks = ["audio_holebase", "audio_gripper"]
    for track in tracks:
        sf.write(os.path.join(trial, track + '.wav'), all_datasets[track], 16000)

if __name__ == "__main__":
    logs = pd.read_csv("episode_times_0220_toy.csv")
    data_folder = "data/test_recordings_0220_toy"
    for idx in range(len(logs)):
        convert_episode(data_folder, logs, idx)