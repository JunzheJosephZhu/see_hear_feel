import sys

from tomlkit import key

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import torch
import numpy as np
# from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti
from svl_project.datasets.imi_dataset_complex import ImitationDatasetFramestackMulti
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_tactile_flow_encoder, make_audio_encoder
from svl_project.models.imi_models import Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T
from torch.autograd import Variable
import shutil
from tqdm import tqdm

def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    # if len_batch > len(batch): # source all the required samples from the original dataset at random
    #     diff = len_batch - len(batch)
    #     for i in range(diff):
    #         batch.append(dataset[np.random.randint(0, len(dataset))])
    return torch.utils.data.dataloader.default_collate(batch)

class MakeVideo():
    def __init__(self, dir=None, framestack=None, name='default', num_episode=1, mods='v_t'):
        self.dir = dir
        self.framestack = framestack
        self.name = name
        self.save_dir = os.path.join(self.dir, self.name)
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)
        self.mydict = {
            'v': {
                'resolution': (160, 120),
                # 'resolution': (160, 120),
                  'font_scale': .5,
                  'thick_scale': 1},
            't': {'resolution': (100, 100),
                  'font_scale': .7,
                  'thick_scale': 1},
        }
        self.path_dict = {'v': [],
                          't': []}
        self.video_writer = {}
        self.frames_ = []
    
    def initialize_sep(self):
        ## save the histogram
        self.hist_dir = os.path.join(self.save_dir, 'hist')
        if os.path.exists(self.hist_dir):
            shutil.rmtree(self.hist_dir)
        os.mkdir(self.hist_dir)
        
        for i in range(self.framestack):
            v_path = os.path.join(self.save_dir, f"v{i}" + '.avi')
            self.video_writer[f"v{i}"] = cv2.VideoWriter(
                v_path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10,
                self.mydict['v']['resolution']
            )
            t_path = os.path.join(self.save_dir, f"t{i}" + '.avi')
            self.video_writer[f"t{i}"] = cv2.VideoWriter(
                t_path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10,
                self.mydict['t']['resolution']
            )
            self.path_dict['v'].append(v_path)
            self.path_dict['t'].append(t_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = 255, 50, 50
        self.thickness = cv2.LINE_8
    
    def save_obs(self, imgs, item, pred=None, gt=None, step=None):
        if item == 'hist':
            imgs.savefig(os.path.join(self.hist_dir, f"{step}.png"))
            return
        for i in range(self.framestack):
            img = imgs[i]
            if item ==  'v':
                img = imgs[i].cpu().permute(1, 2, 0).numpy()
            elif item == 't':
                img = imgs[i].cpu().permute(2, 1, 0).numpy()
            img = np.uint8(img * 255)
            if item == 'v':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img,
                            str(i),
                            (5, 20),
                            self.font,
                            self.mydict[item]['font_scale'],
                            self.color,
                            self.mydict[item]['thick_scale'],
                            self.thickness
                )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.video_writer[item + str(i)].write(img)


def baselineValidate(args):
    print(args.use_flow)
    
    def label_to_action(label):
        action = np.zeros(args.action_dim)
        for d in range(args.action_dim):
            action[args.action_dim - d - 1] = label % 3
            label //= 3
        action -= 1
        return action
            
    def action_to_label(action):
        label = 0
        for d in range(args.action_dim):
            label += 3 ** d * action[args.action_dim - d - 1]
        return label

    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.split('.', 1)[-1] : v for k, v in state_dict.items() if k.startswith(prefix)}
    
    val_csv = pd.read_csv(args.val_csv)
    # val_set = torch.utils.data.ConcatDataset(
    #     [ImitationDatasetFramestackMulti(args.val_csv,
    #                                      args,
    #                                      i,
    #                                      args.data_folder,
    #                                      train=False)
    #      for i in range(min(args.num_episode, len(val_csv)))])
    val_set = ImitationDatasetFramestackMulti(args.val_csv,
                                              args,
                                              args.num_episode,
                                              args.data_folder,
                                              train=False)

    val_loader = DataLoader(val_set, 1, num_workers=4) #, collate_fn=collate_fn
    with torch.no_grad():
        # construct model
        v_encoder = make_vision_encoder(args.embed_dim_v)
        if args.use_flow:
            t_encoder = make_tactile_flow_encoder(args.embed_dim_t)
        else:
            t_encoder = make_tactile_encoder(args.embed_dim_v)
            # t_encoder = make_tactile_encoder(args.embed_dim_t)
        a_encoder = make_audio_encoder()
        v_encoder.eval()
        t_encoder.eval()
        a_encoder.eval()

        actor = Imitation_Actor_Ablation(v_encoder, t_encoder, a_encoder, args)
        # get pretrained parameters
        state_dict = strip_sd(torch.load(args.pretrained)['state_dict'], 'actor.')
        # print("Model's state_dict:")
        # for param_tensor in state_dict:
        #     print(param_tensor, "\t", state_dict[param_tensor].size())
        actor.load_state_dict(state_dict)
        actor.cuda()
        actor.eval()

    cnt = 0
    cor = np.zeros(args.action_dim)
    wrong = np.zeros(args.action_dim)
    total_wrong = 0
    total_cor = 0

    pred_actions = []
    real_actions = []
    pred_labels = []
    real_labels = []
    pred_label_cnts = np.zeros(3 ** args.action_dim)
    real_label_cnts = np.zeros(3 ** args.action_dim)
    
    label_correct = np.zeros(3 ** args.action_dim)
    label_total = np.zeros(3 ** args.action_dim)
    
    debug_info = True
    
    model_dir = '/'.join(args.pretrained.split('/')[:-1])
    
    if args.save_video:
        video_saver = MakeVideo(dir=model_dir, framestack=args.num_stack, name=args.exp_name)
        video_saver.initialize_sep()
    
    for batch in tqdm(val_loader):
        # v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        v_input, t_input, log_spec, keyboard = batch
        v_input = Variable(v_input).cuda()
        t_input = Variable(t_input).cuda()
        a_input = Variable(log_spec).cuda()
        
        v_input.squeeze_(0)
        t_input.squeeze_(0)
        s_v = v_input.shape
        s_t = t_input.shape
        # v_input = torch.reshape(v_input, (s_v[-4]*s_v[-5], 3, s_v[-2], s_v[-1]))
        # t_input = torch.reshape(t_input, (s_t[-4]*s_t[-5], s_t[-3], s_t[-2], s_t[-1]))
        
        ## debugging
        # print(s_v)
        # for i in range(s_v[0]):
        #     img = v_input[i].cpu().permute(1, 2, 0).numpy()
        #     img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        #     cv2.imshow('fixed ' + str(i), img)
        # cv2.waitKey(200)
        
        keyboard = keyboard.numpy()
        action_logits = actor(v_input, t_input, a_input, True).detach().cpu().numpy()
        if args.loss_type == 'cce':
            pred_label = np.argmax(action_logits)
            gt_label = 0
            pred_temp = pred_label
            pred_action = np.zeros(args.action_dim)
            for d in range(args.action_dim):
                gt_label += 3 ** d * keyboard[0][args.action_dim - d - 1]
                pred_action[args.action_dim - d - 1] = pred_temp % 3
                pred_temp //= 3
            # print(f"keyboard {keyboard}, label {gt_label}")
            # print(f"pred_action {pred_action}, label {pred}")
        elif args.loss_type == 'mse':
            pred_action = pred_action.reshape(-1) # * np.array((.003, .003, .0015))
        # keyboard = (keyboard - 1.)#.type(torch.cuda.FloatTensor)
        keyboard = keyboard[0]
        ## debugging
        if debug_info:
            print(f"model output shape {action_logits.shape}")
            print(f"keyboard shape {keyboard.shape}, value {keyboard}")
            print(f"pred shape {pred_action.shape}, value {pred_action}")
            debug_info = False
        ## test exact match
        match = True
        for i in range(args.action_dim):
            if pred_action[i] != keyboard[i]:
                match = False
                break
        if match:
            label_correct[pred_label] += 1
            total_cor += 1
        else:
            total_wrong += 1
        label_total[gt_label] += 1
        ## test partial match
        for i in range(args.action_dim):
            if pred_action[i] == keyboard[i]:
                cor[i] += 1
            else:
                wrong[i] += 1
        pred_actions.append(pred_action)
        real_actions.append(keyboard)
        pred_labels.append(pred_label)
        real_labels.append(gt_label)
        pred_label_cnts[pred_label] += 1
        real_label_cnts[gt_label] += 1
        # print(f"real: {keyboard}, prediction: {pred_action}")
        cnt += 1
        keyboard = keyboard - 1.0
        pred_action = pred_action - 1.0
        
        
        if args.save_video:
            # video_saver.append_frame(v_input, t_input, 'v', pred_action, keyboard)
            video_saver.save_obs(v_input, 'v', pred_action, keyboard, cnt-1)
            video_saver.save_obs(t_input, 't')
            
            ## histogram
            fig, axs = plt.subplots(figsize=(5, 5))
            axs.set_ylim(0, 300)
            axs.plot([0, 81], [0, 0], 'k', linewidth=.5)
            nonzero_labels = list(np.where(pred_label_cnts + real_label_cnts > 0)[0])
            # print(nonzero_labels)
            for label in nonzero_labels:
                if pred_label_cnts[label] > 0 or real_label_cnts[label] > 0:
                    # print(f"plot label {label}, real cnt {real_label_cnts[label]}, pred cnt {pred_label_cnts[label]}")
                    axs.plot([label, label], [0, real_label_cnts[label]], 'b')
                    axs.plot([label+.5, label+.5], [0, pred_label_cnts[label]], 'r')
                    max_label = max(real_label_cnts[label], pred_label_cnts[label])
                    axs.text(x=label, y=-5, s=str(label), fontsize=7)
                    action = label_to_action(label)
                    axs.text(x=label, y=max_label, s=f"{action}", rotation=30, fontsize=7)
            axs.set_title(f"gt {keyboard}, pred {pred_action}, step {cnt-1}")
            axs.set_ylabel('count of occurrence')
            axs.set_xlabel('class')
            
            video_saver.save_obs(fig, 'hist', step=cnt-1)
        
            plt.close(fig)
            

    ## accuracy summary
    acc = cor / (cor + wrong)
    print(f"each direction acc: {acc}")
    acc = total_cor / (total_wrong + total_cor)
    print(f"EM = {acc}")
    print(f"class acc: {label_correct / label_total}")
    predict = np.asarray(pred_actions)
    real = np.asarray(real_actions)
    
    ## plot prediction history
    fig, axs = plt.subplots(args.action_dim + 1, 1, figsize=(15, 10), sharex='col')
    titles = [f"acc: {acc}", 'x', 'y', 'z']
    if args.action_dim == 4:
        titles.append('dz')
    for i in range(len(titles)):
        if i < 1:
            axs[i].plot(real_labels, 'b+', label='real')
            axs[i].plot(pred_labels, 'r.', label='predict')
        else:
            axs[i].plot(real[:, i-1], 'b+', label='real')
            axs[i].plot(predict[:, i-1], 'r.', label='predict')
        axs[i].title.set_text(titles[i])
        axs[i].legend()
    print(args.pretrained.split('/')[:-1] + [f'{args.exp_name}.png'])
    fig.savefig(os.path.join(model_dir, f"{args.exp_name}_label_t.png"))
    
    ## trying better visualization
    fig, axs = plt.subplots()
    axs.plot([0, 81], [0, 0], 'k', linewidth=.5)
    for label in range(3 ** args.action_dim):
        if pred_label_cnts[label] > 0 or real_label_cnts[label] > 0:
            print(f"plot label {label}, real cnt {real_label_cnts[label]}, pred cnt {pred_label_cnts[label]}")
            axs.plot([label, label], [0, real_label_cnts[label]], 'b')
            axs.plot([label+.5, label+.5], [0, pred_label_cnts[label]], 'r')
            max_label = max(real_label_cnts[label], pred_label_cnts[label])
            axs.text(x=label, y=-5, s=str(label), fontsize=7)
            action = label_to_action(label)
            axs.text(x=label, y=max_label, s=f"{action}", rotation=30, fontsize=7)
    axs.set_ylabel('count of occurrence')
    axs.set_xlabel('class')
    axs.set_title(f'acc: {acc}')
    fig.savefig(os.path.join(model_dir, f"{args.exp_name}_labeldist.png"))
    
    plt.show()

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    # p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn_ablation.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--num_workers", default=4, type=int)
    # model
    # p.add("--embed_dim", required=True, type=int)
    p.add("--pretrained", required=True)
    p.add("--freeze_till", required=True, type=int)
    p.add("--num_episode", default=15, type=int)
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--embed_dim_v", required=True, type=int)
    p.add("--embed_dim_t", required=True, type=int)
    p.add("--embed_dim_a", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--loss_type", default="cce")
    p.add("--num_heads", default=8, type=int)
    p.add("--use_flow", default=False, type=bool)
    p.add("--use_mha", default=False)
    p.add("--use_holebase", default=False)
    # data
    p.add("--crop_percent", default=.1, type=float)
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="../data_0401/test_recordings/")
    p.add("--num_camera", required=True, type=int)
    p.add("--total_episode", required=True, type=int)
    p.add("--ablation", required=True)
    p.add("--use_layernorm", default=False, type=bool)
    p.add("--exp_name", default=None)
    p.add("--save_video", default=False, action='store_true')

    args = p.parse_args()
    baselineValidate(args)