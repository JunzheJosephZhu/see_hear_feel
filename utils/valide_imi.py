from email.policy import default
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import torch
import numpy as np
from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti, ImitationOverfitDataset, ImitationDatasetFramestack
from svl_project.models.encoders import make_vision_encoder, make_tactile_encoder, make_tactile_flow_encoder, make_audio_encoder
from svl_project.models.imi_models import Imitation_Actor_Ablation
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T
from torch.autograd import Variable


def baselineValidate(args):
    print(args.use_flow)
    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.split('.', 1)[-1] : v for k, v in state_dict.items() if k.startswith(prefix)}
        # return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    
    val_csv = pd.read_csv(args.val_csv)
    # val_set = torch.utils.data.ConcatDataset(
    #     [ImitationOverfitDataset(args.val_csv, i, args.data_folder) for i in range(len(val_csv))])
    val_set = torch.utils.data.ConcatDataset(
        [ImitationDatasetFramestackMulti(args.val_csv,
                                         args,
                                         i,
                                         args.data_folder,
                                         train=False)
         for i in range(min(args.num_episode, len(val_csv)))])

    val_loader = DataLoader(val_set, 1, num_workers=8)
    with torch.no_grad():
        # construct model
        v_encoder = make_vision_encoder(args.embed_dim_v)
        if args.use_flow:
            t_encoder = make_tactile_flow_encoder(args.embed_dim_t)
        else:
            t_encoder = make_tactile_encoder(args.embed_dim_t)
        a_encoder = make_audio_encoder(args.conv_bottleneck, args.embed_dim_a)
        v_encoder.eval()
        t_encoder.eval()
        a_encoder.eval()

        actor = None
        actor = Imitation_Actor_Ablation(v_encoder, t_encoder, a_encoder, args)
        # get pretrained parameters
        state_dict = strip_sd(torch.load(args.pretrained)['state_dict'], 'actor.')
        print("Model's state_dict:")
        for param_tensor in state_dict:
            print(param_tensor, "\t", state_dict[param_tensor].size())
        actor.load_state_dict(state_dict)
        actor.cuda()
        actor.eval()

    cnt = 0
    cor = np.array([0, 0, 0])
    wrong = np.array([0, 0, 0])
    total_wrong = 0
    total_cor = 0

    predict = []
    real = []

    for batch in val_loader:
        # v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        v_input, t_input, log_spec, keyboard = batch
        v_input = Variable(v_input).cuda()
        t_input = Variable(t_input).cuda()
        a_input = Variable(log_spec).cuda()
        
        s_v = v_input.shape
        # print(f"s_v {s_v}")
        s_t = t_input.shape
        # print(f"s_t {s_t}")
        v_input.squeeze_(0)
        t_input.squeeze_(0)
        # v_input = torch.reshape(v_input, (s_v[-4]*s_v[-5], 3, s_v[-2], s_v[-1]))
        # t_input = torch.reshape(t_input, (s_t[-4]*s_t[-5], s_t[-3], s_t[-2], s_t[-1]))
        
        # v_total, keyboard = v_total[0], keyboard[0]
        # v_gripper, v_fixed = v_total[0], v_total[1]
        # cv2.imshow('gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('after gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('fixed', v_input.squeeze().cpu().permute(1, 2, 0).numpy())
        # cv2.waitKey(200)
        keyboard = keyboard.numpy()
        # action_pred = actor(v_gripper_inp, v_fixed_inp, True).detach().numpy()
        pred_action = actor(v_input, t_input, a_input, True).detach().cpu().numpy()
        if args.loss_type == 'cce':
            # pred_action = pred_action.reshape(3, -1)
            # pred_action = (np.argmax(pred_action, axis=1) - 1) * np.array((.003, .003, .0015))
            x = np.argmax(pred_action) // 9
            y = (np.argmax(pred_action) - x * 9) // 3
            z = int(np.argmax(pred_action) - x * 9 - y * 3)
            # print((x, y, z))
            # pred_action = (np.array((x, y, z)) - 1.) * np.array((.003, .003, .0015))
            pred_action = np.array((x, y, z))
            # print(pred_action)
        elif args.loss_type == 'mse':
            pred_action = pred_action.reshape(-1) # * np.array((.003, .003, .0015))
        # keyboard = (keyboard - 1.)#.type(torch.cuda.FloatTensor)
        # print(keyboard.shape)
        match = True
        for i in range(3):
            if pred_action[i] != keyboard[0][i]:
                match = False
                break
        if match:
            total_cor += 1
        else:
            total_wrong += 1
        for i in range(3):
            if pred_action[i] == keyboard[0][i]:
                cor[i] += 1
            else:
                wrong[i] += 1
        predict.append(pred_action)
        real.append(keyboard)
        # print(f"real: {keyboard}, prediction: {pred_action}")
        cnt += 1
        # if cnt == 150:
        #     break
    # print(f"{cnt} steps in total.")
    ## accuracy summary
    acc = cor / (cor + wrong)
    print(f"each direction acc: {acc}")
    acc = total_cor / (total_wrong + total_cor)
    print(f"EM = {acc}")
    predict = np.asarray(predict)
    real = np.asarray(real)
    fig, axs = plt.subplots(3, 1, sharex='col')
    # print(real.shape)
    legends = ['x', 'y', 'z']
    for i in range(len(legends)):
        axs[i].plot(real[:, 0, i], 'b+', label='real')
        axs[i].plot(predict[:, i], 'rx', label='predict')
        axs[i].legend()
    plt.show()
    # fig = plt.figure(0)
    # plt.title("x")
    # plt.scatter(range(cnt),predict[:, 0], s = 0.3)
    # plt.scatter(range(cnt),real[:, 0], s = 0.3, alpha= 0.5)
    # plt.xlabel("count of batches")
    # plt.ylabel("actions(0:move -;1:stay;2:move +")
    # plt.legend(["pred","real"])
    # fig = plt.figure(1)
    # plt.title("y")
    # plt.scatter(range(cnt), predict[:, 1], s = 0.3)
    # plt.scatter(range(cnt), real[:, 1],s = 0.3,alpha= 0.5)
    # plt.xlabel("count of batches")
    # plt.ylabel("actions(0:move -;1:stay;2:move +")
    # plt.legend(["pred", "real"])
    # fig = plt.figure(2)
    # plt.title("z")
    # plt.scatter(range(cnt), predict[:, 2],s = 0.3)
    # plt.scatter(range(cnt), real[:, 2],s = 0.3,alpha=0.5)
    # plt.xlabel("count of batches")
    # plt.ylabel("actions(0:move -;1:stay;2:move +")
    # plt.legend(["pred", "real"])
    # plt.show()



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
    p.add("--num_episode", default=20, type=int)
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
    # data
    p.add("--crop_percent", default=.1, type=float)
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--train_csv", default="train_0331.csv")
    p.add("--val_csv", default="val_0331.csv")
    p.add("--data_folder", default="../data_0318/test_recordings/")
    p.add("--num_camera", required=True, type=int)
    p.add("--total_episode", required=True, type=int)
    p.add("--ablation", required=True)

    args = p.parse_args()
    baselineValidate(args)