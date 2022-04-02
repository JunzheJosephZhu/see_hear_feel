import sys

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
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T
from torch.autograd import Variable


def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    # if len_batch > len(batch): # source all the required samples from the original dataset at random
    #     diff = len_batch - len(batch)
    #     for i in range(diff):
    #         batch.append(dataset[np.random.randint(0, len(dataset))])
    return torch.utils.data.dataloader.default_collate(batch)

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

    val_loader = DataLoader(val_set, 1, num_workers=4) #, collate_fn=collate_fn
    with torch.no_grad():
        # construct model
        v_encoder = make_vision_encoder(args.embed_dim_v)
        if args.use_flow:
            t_encoder = make_tactile_flow_encoder(args.embed_dim_t)
        else:
            t_encoder = make_tactile_encoder(args.embed_dim_v)
            # t_encoder = make_tactile_encoder(args.embed_dim_t)
        a_encoder = make_audio_encoder(args.embed_dim_a)
        v_encoder.eval()
        t_encoder.eval()
        a_encoder.eval()

        actor = None
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

    predict = []
    real = []
    predict_label = []
    real_label = []
    label_correct = np.zeros(3 ** args.action_dim)
    label_total = np.zeros(3 ** args.action_dim)
    
    debug_info = True
    for batch in val_loader:
        # v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        v_input, t_input, log_spec, keyboard = batch
        v_input = Variable(v_input).cuda()
        t_input = Variable(t_input).cuda()
        a_input = Variable(log_spec).cuda()
        
        v_input.squeeze_(0)
        t_input.squeeze_(0)
        # s_v = v_input.shape
        # s_t = t_input.shape
        # v_input = torch.reshape(v_input, (s_v[-4]*s_v[-5], 3, s_v[-2], s_v[-1]))
        # t_input = torch.reshape(t_input, (s_t[-4]*s_t[-5], s_t[-3], s_t[-2], s_t[-1]))
        
        ## debugging
        # v_total, keyboard = v_total[0], keyboard[0]
        # v_gripper, v_fixed = v_total[0], v_total[1]
        # cv2.imshow('gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('after gripper', v_gripper.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow('fixed', v_input.squeeze().cpu().permute(1, 2, 0).numpy())
        # cv2.waitKey(200)
        keyboard = keyboard.numpy()
        action_logits = actor(v_input, t_input, a_input, True).detach().cpu().numpy()
        if args.loss_type == 'cce':
            pred = np.argmax(action_logits)
            gt_label = 0
            pred_temp = pred
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
        ## debugging
        if debug_info:
            print(f"model output shape {action_logits.shape}")
            print(f"keyboard shape {keyboard.shape}, value {keyboard}")
            print(f"pred shape {pred_action.shape}, value {pred_action}")
            debug_info = False
        ## test exact match
        match = True
        for i in range(args.action_dim):
            if pred_action[i] != keyboard[0][i]:
                match = False
                break
        if match:
            label_correct[pred] += 1
            total_cor += 1
        else:
            total_wrong += 1
        label_total[gt_label] += 1
        ## test partial match
        for i in range(args.action_dim):
            if pred_action[i] == keyboard[0][i]:
                cor[i] += 1
            else:
                wrong[i] += 1
        predict.append(pred_action)
        real.append(keyboard)
        predict_label.append(pred)
        real_label.append(gt_label)
        # print(f"real: {keyboard}, prediction: {pred_action}")
        cnt += 1

    ## accuracy summary
    acc = cor / (cor + wrong)
    print(f"each direction acc: {acc}")
    acc = total_cor / (total_wrong + total_cor)
    print(f"EM = {acc}")
    print(f"class acc: {label_correct / label_total}")
    predict = np.asarray(predict)
    real = np.asarray(real)
    
    ## plot distribution
    fig, axs = plt.subplots(args.action_dim + 1, 1, figsize=(15, 10), sharex='col')
    titles = ['labels', 'x', 'y', 'z']
    if args.action_dim == 4:
        titles.append('dz')
    for i in range(len(titles)):
        if i < 1:
            axs[i].plot(real_label, 'b+', label='real')
            axs[i].plot(predict_label, 'r.', label='predict')
        else:
            axs[i].plot(real[:, 0, i-1], 'b+', label='real')
            axs[i].plot(predict[:, i-1], 'r.', label='predict')
        axs[i].title.set_text(titles[i])
        axs[i].legend()
    print(args.pretrained.split('/')[:-1] + ['validation.png'])
    plt.savefig('/'.join(args.pretrained.split('/')[:-1] + ['validation.png']))
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
    p.add("--use_holebase", default=False)
    # data
    p.add("--crop_percent", default=.1, type=float)
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="../data_0331/test_recordings/")
    p.add("--num_camera", required=True, type=int)
    p.add("--total_episode", required=True, type=int)
    p.add("--ablation", required=True)
    p.add("--use_layernorm", default=False, type=bool)

    args = p.parse_args()
    baselineValidate(args)