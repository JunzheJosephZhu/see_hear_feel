from email.mime import audio
import sys
from turtle import width

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

tstamp = '2022-05-02 20:40:16.687281'
DIR = '../test_recordings/' + tstamp
# DIR = '../data_0424/test_recordings/' + tstamp
# f = h5py.File(os.path.join(DIR, 'data.hdf5'), 'r')
# action_path = os.path.join(DIR, 'timestamp.json')
action_hist = None
# if os.path.exists(action_path):
    # action_hist_file = pd.read_csv(action_path)
    # action_hist = action_hist_file['action_history']
# with open(os.path.join(DIR, "timestamps.json")) as ts:
#     action_hist_file = json.load(ts)

# print(action_hist)
# print(action_hist[0].tolist()[0])
# tstamp = '2022-03-31 20:55:52.238120'
# f = h5py.File('./data_0331/test_recordings/' + tstamp + '/data.hdf5', 'r')
item_list = {
    1: 'cam_gripper_color',
    2: 'cam_fixed_color',
    3: 'left_gelsight_frame',
    4: 'left_gelsight_flow',
    5: 'audio_holebase_left',
    6: 'audio_gripper_left',
    7: 'audio_gripper_right',
    8: 'audio_holebase_right',
    9: 'confusion_matrix'
}
test_items = [2,3,5,9]
ablation = 'v_t_a'


class Tests():
    def __init__(self, args):
        self.dir = os.path.join('../test_recordings', tstamp)
        # self.dir = os.path.join('../data_0424/test_recordings', tstamp)
        self.video = args.video
        if args.video and (not args.store):
            self.make_video()
            return
        self.store = args.store
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = 0, 255, 255
        self.thickness = cv2.LINE_8
        for i in test_items:
            self.test_item = item_list[i]
            print(f"{self.test_item} shape: {f[self.test_item].shape}")
            self.path = os.path.join(self.dir, self.test_item)
            if self.test_item in ['cam_gripper_color', 'cam_fixed_color', 'left_gelsight_frame']:
                self.test_img(os.path.join(self.dir, self.test_item))
            elif self.test_item == 'left_gelsight_flow':
                self.test_flow()
            elif self.test_item.startswith('audio'):
                self.test_audio()
            elif self.test_item == 'confusion_matrix':
                print('con')
                dir = os.path.join(self.dir, 'confusion_matrix')
                if not self.video:
                    if os.path.exists(dir):
                        shutil.rmtree(dir)
                    os.mkdir(dir)
                print('dir')
                self.test_confusion_matrix(dir)
        # plt.show()
        f.close()
        # if args.video:
        #     self.make_video()
    
    def test_confusion_matrix(self, dir):
        figsize = 3, 2
        if self.video and not self.store:
            print('video')
            resolution = 300, 200
            video_writer = cv2.VideoWriter(
                self.path + '.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                resolution#, False
            )
            figs = os.listdir(dir)
            print(figs[:10])
            # figs.sort()
            # print(figs[:10])
            for idx in tqdm(range(len(figs))):
                # print('xxx')
                img_path = os.path.join(dir, f"{idx}.png")
                img_read = cv2.imread(img_path)
                cv2.imshow('read', img_read)
                cv2.waitKey(1)
                # print(img_read)
                # img_read = np.uint8(img_read * 255)
                # print(img_read.shape)
                # img_read = img_read[:, :, :3]
                # img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
                video_writer.write(img_read)
            return
        cnt = 0
        # if self.store:
        #     video_writer = cv2.VideoWriter(
        #         self.path + '.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
        #         (figsize[0] * 100, figsize[1] * 100), False
        #     )
        for s in tqdm(f[self.test_item].iter_chunks()):
            weights = f[self.test_item][s]
            # print(f"weights {weights} {weights.shape}")
            modalities = ablation.split('_')
            use_vision = 'v' in modalities
            use_tactile = 't' in modalities
            use_audio = 'a' in modalities
            used_input = []
            output = []
            if use_vision:
                used_input.append('v_in')
                output.append('v_out')
            if use_tactile:
                used_input.append('t_in')
                output.append('t_out')
            if use_audio:
                used_input.append('a_in')
                output.append('a_out')
            df_cm = pd.DataFrame(weights, index = output, columns=used_input)
            
            plt.figure(figsize=figsize)
            sn.set(font_scale=1)
            img = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", annot_kws={"fontsize":10}).get_figure()
            img.savefig(os.path.join(dir, f"{cnt}.png"))
            plt.close(img)
            # img = plt.imread(os.path.join(dir, f"{cnt}.png"))
            # img = np.uint8(img * 255)
            # img = img[:, :, :3]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.store:
                pass
                # video_writer.write(img)
            else:
                cv2.imshow(self.test_item, img)
                cv2.waitKey(100)
            cnt += 1

    def test_audio(self):
        audio_buffer = f[self.test_item]
        if self.store:
            import soundfile as sd
            fs = 44100
            sd.write(
                file=self.path + '.wav',
                data=audio_buffer[:],
                samplerate=fs
            )
        else:
            # if self.test_item == 'audio_holebase_right':
            plt.figure()
            audio_buffer_arr = audio_buffer[:]
            x_lim = np.arange(0, len(audio_buffer)) / 4410
            audio_buffer_arr = np.abs(audio_buffer[:])
            audio_buffer_arr = np.clip(audio_buffer_arr,
                                       a_min=0,
                                       a_max=0.5)
            plt.plot(x_lim, audio_buffer_arr)
            plt.title(self.test_item)
            plt.show()
            # step = int(len(audio_buffer_arr)/44100)
            # for i in range(step):
            #     plt.figure(i)
            #     spec_nomel = torch.fft.rfft(torch.tensor(audio_buffer_arr[i*44100:(i+1)*44100]).type(torch.FloatTensor))
            #     x = torch.fft.rfftfreq(len(audio_buffer_arr[i*44100:(i+1)*44100]), 1 / 44100)
            #     plt.plot(x[:10000], np.abs(spec_nomel[:10000]))
            #     plt.show()                      a_max=0.5)
            plt.plot(x_lim, audio_buffer_arr)
            plt.title(self.test_item)
            plt.show

        plt.savefig(f"{self.path}.png")

    def test_img(self, dir):
        if self.test_item == 'left_gelsight_frame':
            resolution = 300, 400  # 400, 300
            font_scale = .7
            thick_scale = 1
        else:
            resolution = 320, 240
            font_scale = 0.7
            thick_scale = 1
        if self.store:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            video_writer = cv2.VideoWriter(
                self.path + '.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, resolution
            )
        cnt = 0
        for s in f[self.test_item].iter_chunks():
            img = f[self.test_item][s]
            if self.test_item == 'left_gelsight_frame':
                img = np.transpose(img, (1, 0, 2))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if self.test_item == 'cam_fixed_color':
            #     cx, cy = 200, 105
            #     action_curr = np.array(action_hist[cnt])
            #     print(action_curr)
            #     print(action_curr[0])
            #     action = np.sign(action_curr[:3])
            #     cnt += 1
            #     print(action)
            #     dx, dy, dz = int(action[0]), int(action[1]), int(action[2])
            #     cv2.circle(img,
            #                center=(cx, cy),
            #                radius=1,
            #                color=(0, 0, 0))
            #     x0 = 30
            #     dir_x = np.deg2rad(-60)
            #     nxx, nxy = np.sin(dir_x), np.cos(dir_x)
            #     print(nxx, nxy)
            #     cv2.arrowedLine(img,
            #                     (cx + int(dx * x0 * nxx), cy + int(dx * x0 * nxy)),
            #                     (cx + int(dx * (x0 + 20) * nxx), cy + int(dx * (x0 + 20) * nxy)),
            #                     color=(10, 10, 255),
            #                     thickness=2,
            #                     tipLength=0.3)
            #     y0 = 30
            #     dir_y = np.deg2rad(240)
            #     nyx, nyy = np.sin(dir_y), np.cos(dir_y)
            #     print(nyx, nyy)
            #     cv2.arrowedLine(img,
            #                     (cx + int(dy * y0 * nyx), cy + int(dy * y0 * nyy)),
            #                     (cx + int(dy * (y0 + 20) * nyx), cy + int(dy * (y0 + 20) * nyy)),
            #                     color=(10, 255, 10),
            #                     thickness=2,
            #                     tipLength=0.3)
            #     ## z
            #     cv2.arrowedLine(img,
            #                     (10, 120),
            #                     (10, 120 - dz * 20),
            #                     color=(255, 255, 255),
            #                     thickness=2,
            #                     tipLength=0.3)
            # print(img)
            if self.test_item == 'cam_fixed_color' and action_hist != None:
                str = f"action: {action_hist[cnt]}"
            else:
                str = self.test_item
            cv2.putText(img,
                        str, (5, 50),
                        # self.test_item, (50, 50),
                        self.font, font_scale/2, self.color, thick_scale, self.thickness
                        )
            if self.store:
                if self.test_item == 'left_gelsight_frame':
                    cv2.imwrite(os.path.join(dir, f"{cnt}.png"), img)
                video_writer.write(img)
            else:
                cv2.imshow(self.test_item, img)
                cv2.waitKey(100)
            cnt += 1

    def test_flow(self):
        cnt = 0
        resolution = 300, 400  # 400, 300
        # resolution = 400, 300
        font_scale = .5
        thick_scale = 1
        if self.store:
            video_writer = cv2.VideoWriter(
                self.path + '.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, resolution, False
            )
        for s in f[self.test_item].iter_chunks():
            cnt += 1
            img = np.ones((300, 400))
            flow = f[self.test_item][s]
            for i in range(flow.shape[1]):
                for j in range(flow.shape[2]):
                    pt1 = (int(flow[0, i, j]), int(flow[1, i, j]))
                    pt2 = (int(flow[2, i, j]), int(flow[3, i, j]))
                    cv2.arrowedLine(img, pt1, pt2, (0, 0, 0))
            img = np.uint8(img * 255)
            img = np.transpose(img)
            img = np.ascontiguousarray(img)
            # print(img)
            cv2.putText(img,
                        self.test_item, (50, 50),
                        self.font, font_scale, self.color, thick_scale, self.thickness
                        )
            if self.store:
                video_writer.write(img)
            else:
                cv2.imshow(self.test_item, img)
                cv2.waitKey(100)

    def make_video(self):
        import moviepy.editor as mpe
        final_video = None
        clip_dict = {}
        for item_idx in test_items:#item_list.keys():
            item = item_list[item_idx]
            # these are videos
            if item_idx <= 4:
                clip_dict[item] = mpe.VideoFileClip(os.path.join(self.dir, item + '.avi'))
                if item_idx <= 2:
                    clip_dict[item] = clip_dict[item].resize(height=400)
            # # and these are audios
            elif item_idx == 9:
                clip_dict[item] = mpe.VideoFileClip(os.path.join(self.dir, item + '.avi'))
            else:
                clip_dict[item] = mpe.AudioFileClip(os.path.join(self.dir, item + '.wav'))

                
        
        # broken gelsight flow
        comp_clip = mpe.clips_array([
            [clip_dict['cam_fixed_color'], 
             clip_dict['confusion_matrix'], 
             clip_dict['left_gelsight_frame']]
        ])

        # comp_clip = mpe.clips_array([
        #     [clip_dict['cam_fixed_color'], clip_dict['cam_gripper_color']],
        #     [clip_dict['left_gelsight_frame'], clip_dict['left_gelsight_flow']]
        #     # [clip_dict['cam_gripper_color'], clip_dict['left_gelsight_frame']]
        # ])
        # comp_clip = comp_clip.resize(width=480)
        
        final_video = comp_clip.set_audio(clip_dict['audio_holebase_left'])
        final_video.write_videofile(os.path.join(self.dir, "final_video_holebase.mp4"))


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
        # comp_clip = mpe.clips_array([[v_list[0], t_list[0]]])
        # comp_clip = mpe.clips_array([[v_list[0], v_list[1]],
        #                             [t_list[0], t_list[1]],
        #                             [v_list[2], v_list[3]],
        #                             [t_list[2], t_list[3]]])
        comp_clip = mpe.clips_array([[v_list[0], t_list[0]],
                                    [v_list[1], t_list[1]],
                                    [v_list[2], t_list[2]]])
        comp_clip.write_videofile(os.path.join(self.dir, "vt.mp4"))

    def save_log_video(self, items=None):
        paths = {}
        clips = {}
        if items is None:
            items = ['arrow', 'seq']
        clip_arr = []
        for item in items:
            paths[item] = os.path.join(self.dir, item + '.avi')
            clips[item] = mpe.VideoFileClip(paths[item])
            clip_arr.append([clips[item]])
        # comp_clip = mpe.clips_array([[clips['arrow']],
        #                              [clips['seq']]])
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
    p.add_argument("--store", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--data_folder", default='../testing_models/pour/ablation_0429_0430/imi_learn_ablation_vf_t_a_pour_mha/checkpoints/exp1')
    args = p.parse_args()

    ## save histogram video demo
    analyze_video = Analysis(args.data_folder, ['seq', 'audio','confusion']) #'audio','confusion'
    
    # ## compose validation videos
    analyze_video.save_log_video(items=['seq', 'audio','confusion']) #'audio', 'confusion'

    # # # ## add hist video
    analyze_video.add_log_video()

    ## compose human demo / data
    # Tests(args)