import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import torch
import numpy as np
from svl_project.datasets.imi_dataset import ImitationOverfitDataset, ImitationDatasetFramestack
from svl_project.models.encoders import make_vision_encoder
from svl_project.models.imi_models import Imitation_Baseline_Actor_Tuning
from svl_project.engines.imi_engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T


def baselineValidate(args):
    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    
    device = torch.device('cuda')

    val_csv = pd.read_csv(args.val_csv)
    # val_set = torch.utils.data.ConcatDataset(
    #     [ImitationOverfitDataset(args.val_csv, i, args.data_folder) for i in range(len(val_csv))])
    val_set = torch.utils.data.ConcatDataset(
        [ImitationDatasetFramestack(args.val_csv, args, i, device, args.data_folder) for i in range(len(val_csv))])

    val_loader = DataLoader(val_set, 1, num_workers=8)
    with torch.no_grad():
        # construct model
        v_encoder = make_vision_encoder(args.conv_bottleneck, args.embed_dim, (2, 2))#, int(args.num_stack * 3 * 2))
        actor = None
        actor = Imitation_Baseline_Actor_Tuning(v_encoder, args)
        # get pretrained parameters
        state_dict = strip_sd(torch.load(args.pretrained)['state_dict'], 'actor.')
        print("Model's state_dict:")
        for param_tensor in state_dict:
            print(param_tensor, "\t", state_dict[param_tensor].size())
        actor.load_state_dict(state_dict)
        actor.to(device)
        actor.eval()

    cnt = 0
    cor = np.array([0, 0, 0])
    wrong = np.array([0, 0, 0])

    predict = []
    real = []

    for batch in val_loader:
        # v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        v_total, keyboard = batch
        # print(batch[0])
        # v_total, keyboard = v_total[0], keyboard[0]
        # v_gripper, v_fixed = v_total[0], v_total[1]
        # cv2.imshow('gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('after gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('fixed', v_fixed.cpu().permute(1, 2, 0).numpy())
        # cv2.waitKey(1)
        keyboard = keyboard.numpy()
        # action_pred = actor(v_gripper_inp, v_fixed_inp, True).detach().numpy()
        pred_action = actor(v_total, True).detach().cpu().numpy()
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
        keyboard = (keyboard - 1.)#.type(torch.cuda.FloatTensor)
        for i in range(3):
            if pred_action[i] == keyboard[i]:
                cor[i] += 1
            else:
                wrong[i] += 1
        # predict.append(pred_action)
        # real.append(keyboard)
        # print(f"real: {keyboard}, prediction: {pred_action}")
        cnt += 1
        # if cnt == 150:
        #     break
    # print(f"{cnt} steps in total.")
    # predict = np.asarray(predict)
    # real = np.asarray(real)
    fig, axs = plt.subplots(3, 1, sharex='col')
    legends = ['x', 'y', 'z']
    for i in range(len(legends)):
        axs[i].plot(real[:, 0], 'b+', label='real')
        axs[i].plot(predict[:, 0], 'rx', label='predict')
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
    # acc = cor / (cor + wrong)
    # print(acc)

# def baselineRegValidate(args):
#     def strip_sd(state_dict, prefix):
#         """
#         strip prefix from state dictionary
#         """
#         return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

#     # get pretrained model
#     val_set = ImmitationDataSet(args.val_csv)
#     val_loader = DataLoader(val_set, 1, num_workers=0)
#     ckpt_path = "last.ckpt"
#     v_gripper_encoder = make_vision_encoder(args.embed_dim)
#     v_fixed_encoder = make_vision_encoder(args.embed_dim)

#     actor = Immitation_Baseline_Actor(
#         v_gripper_encoder, v_fixed_encoder, args.embed_dim, args.action_dim)
#     state_dict = strip_sd(torch.load(ckpt_path)['state_dict'], 'actor.')
#     print("Model's state_dict:")
#     for param_tensor in state_dict:
#         print(param_tensor, "\t", state_dict[param_tensor].size())
#     actor.load_state_dict(state_dict)
#     actor.eval()
#     cnt = 0
#     cor = np.array([0, 0, 0])
#     wrong = np.array([0, 0, 0])

#     predict = []
#     real = []

#     for batch in val_loader:
#         v_gripper_inp, v_fixed_inp, _, _, keyboard, _ = batch
#         action_pred = actor(v_gripper_inp, v_fixed_inp, True).detach().numpy()
#         action_pred = action_pred.reshape(3)
#         # action_pred = (np.argmax(action_pred, axis=1))
#         keyboard = (keyboard.numpy() - 1)
#         # keyboard[0][:2] = keyboard[0][:2] * 0.006
#         # keyboard[0][2] = keyboard[0][2] * 0.003
#         # for i in range(3):
#         #     if(action_pred[i] == keyboard[0][i]):
#         #         cor[i] += 1
#         #     else:
#         #         wrong[i] += 1
#         predict.append(action_pred)
#         real.append(keyboard)

#         cnt += 1
#         # if cnt == 100:
#         #     break

#     predict = np.asarray(predict)
#     real = np.asarray(real).reshape(cnt,3)
#     fig = plt.figure(0)
#     plt.title("x")
#     plt.scatter(range(cnt),predict[:, 0], s = 0.3)
#     plt.scatter(range(cnt),real[:, 0], s = 0.3, alpha= 0.5)
#     plt.xlabel("count of batches")
#     plt.ylabel("actions(0:move -;1:stay;2:move +")
#     plt.legend(["pred","real"])
#     fig = plt.figure(1)
#     plt.title("y")
#     plt.scatter(range(cnt), predict[:, 1], s = 0.3)
#     plt.scatter(range(cnt), real[:, 1],s = 0.3,alpha= 0.5)
#     plt.xlabel("count of batches")
#     plt.ylabel("actions(0:move -;1:stay;2:move +")
#     plt.legend(["pred", "real"])
#     fig = plt.figure(2)
#     plt.title("z")
#     plt.scatter(range(cnt), predict[:, 2],s = 0.3)
#     plt.scatter(range(cnt), real[:, 2],s = 0.3,alpha=0.5)
#     plt.xlabel("count of batches")
#     plt.ylabel("actions(0:move -;1:stay;2:move +")
#     plt.legend(["pred", "real"])
#     plt.show()
#     # acc = cor / (cor + wrong)
#     # print(acc)

# def compute_loss_cee(pred, demo):
#     """
#     pred: # [batch, 3 * action_dims]
#     demo: # [batch, action_dims]
#     """
#     batch_size = pred.size(0)
#     space_dim = demo.size(-1)
#     # [batch, 3, num_dims]
#     pred = pred.reshape(batch_size, 3, space_dim)
#     return torch.nn.CrossEntropyLoss(pred, demo)

# def compute_loss_mse(pred, demo):
#     """
#     pred: # [batch, 3 * action_dims]
#     demo: # [batch, action_dims]
#     """
#     batch_size = pred.size(0)
#     space_dim = demo.size(-1)
#     # [batch, 3, num_dims]
#     pred = pred.reshape(batch_size, space_dim)
#     return torch.nn.MSELoss(pred, demo)


if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int)
    # model
    p.add("--embed_dim", required=True, type=int)
    p.add("--pretrained", required=True)
    p.add("--freeze_till", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--num_heads", type=int)
    p.add("--loss_type", required=True)
    p.add("--num_stack", default=4, type=int)
    p.add("--frameskip", default=3, type=int)
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--crop_percent", default=.1, type=float)
    p.add("--resized_height", required=True, type=int)
    p.add("--resized_width", required=True, type=int)    
    p.add("--num_episode", default=10, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="../data_0214/test_recordings/")
    


    args = p.parse_args()
    baselineValidate(args)