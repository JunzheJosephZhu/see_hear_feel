import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import moviepy.editor as mpe
from tqdm import tqdm
import json
import pandas as pd
import torch
import shutil
import seaborn as sn


class Analysis():
    def __init__(self, dir=None, items=None):
        self.dir = dir
        if items is None:
            self.items = ['hist', 'seq', 'arrow', 'audio', 'confusion']
        else:
            self.items = items
        self._save_vt_video()
        for item in self.items:
            self._save_video_from(item)
    
    def _save_video_from(self, item=None):
        img_dir = os.path.join(self.dir, item)
        print(img_dir)
        imgs = os.listdir(img_dir)
        min_idx = 0
        max_idx = len(imgs)
        if item == 'hist':
            resolution = 500, 500
        elif item == 'seq':
            resolution = 600, 400
        elif item == 'arrow':
            resolution = 400, 200
        elif item == 'audio':
            resolution = 600, 300 #640, 480
        elif item == 'confusion':
            resolution = 500, 400
        video_writer = cv2.VideoWriter(
            os.path.join(self.dir, item + '.avi'),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, resolution  # , False
        )
        for idx in tqdm(range(min_idx, max_idx)):
            img_path = os.path.join(img_dir, f"{str(idx)}.png")
            img_read = plt.imread(img_path)
            img_read = np.uint8(img_read * 255)
            img_read = img_read[:, :, :3]
            img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
            video_writer.write(img_read)


    def _save_vt_video(self):
        items = os.listdir(self.dir)
        logs = ['hist', 'seq', 'arrow', 'audio', 'confusion']
        for log in logs:
            if log in items:
                items.remove(log)
        items.sort()
        print(items)
        v_list = []
        t_list = []
        for item in items:
            print(item)
            clip = mpe.VideoFileClip(os.path.join(self.dir, item))
            if item[0] == 'v':
                v_list.append(clip)
            elif item[0] == 't':
                t_list.append(clip)
        comp_clip = mpe.clips_array([[v_list[0], t_list[0]],
                                    [v_list[1], t_list[1]],
                                    [v_list[2], t_list[2]]])
        comp_clip.write_videofile(os.path.join(self.dir, "vt.mp4"))

    def save_log_video(self, items=None):
        paths = {}
        clips = {}
        clip_arr = []
        for item in self.items:
            paths[item] = os.path.join(self.dir, item + '.avi')
            clips[item] = mpe.VideoFileClip(paths[item])
            clip_arr.append([clips[item]])
        comp_clip = mpe.clips_array(clip_arr)
        comp_clip.write_videofile(os.path.join(self.dir, "logs.mp4"))

    def add_log_video(self):
        logs_path = os.path.join(self.dir, 'logs.mp4')
        vt_path = os.path.join(self.dir, 'vt.mp4')

        logs = mpe.VideoFileClip(logs_path)  # .resize(height=500)
        vt = mpe.VideoFileClip(vt_path).resize(width=1080, height=705)
        comp_clip = mpe.clips_array([[vt, logs]])
        comp_clip.write_videofile(os.path.join(self.dir, "vt_logs.mp4"))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", default=None)
    args = p.parse_args()

    ## save histogram video demo
    analyze_video = Analysis(args.data_folder, ['seq', 'audio']) #'audio','confusion'
    
    ## compose validation videos
    analyze_video.save_log_video(items=['seq', 'audio']) #'audio', 'confusion'

    ## add hist video
    analyze_video.add_log_video()

    ## compose human demo / data
    # Tests(args)