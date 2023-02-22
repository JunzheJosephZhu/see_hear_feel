from email.policy import default
from re import L
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import torch
import numpy as np

# from svl_project.datasets.imi_dataset import ImitationDatasetFramestackMulti
from src.datasets.imi_dataset_complex import ImitationDatasetFramestackMulti
from src.models.encoders import (
    make_vision_encoder,
    make_tactile_encoder,
    make_tactile_flow_encoder,
    make_audio_encoder,
)
from src.models.imi_models import Actor
from src.engines.engine import ImiBaselineLearn_Tuning
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from torchvision import transforms as T
from torch.autograd import Variable
import shutil
from tqdm import tqdm
from save_val_obs import MakeVideo
import seaborn as sn

# keyboard[keyboard == 0] = 0
# keyboard[keyboard == 3] = 1
# keyboard[keyboard == 4] = 2
# keyboard[keyboard == 5] = 2
# keyboard[keyboard == 6] = 3

pouring_action_map = {0: 0, 1: 3, 2: 5, 3: 6}

key_insertion_action_map = {
    0: [0, 1, 1],
    1: [2, 1, 1],
    2: [1, 0, 1],
    3: [1, 2, 1],
    4: [1, 1, 0],
    5: [1, 1, 2],
}


def collate_fn(batch):
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    # if len_batch > len(batch): # source all the required samples from the original dataset at random
    #     diff = len_batch - len(batch)
    #     for i in range(diff):
    #         batch.append(dataset[np.random.randint(0, len(dataset))])
    return torch.utils.data.dataloader.default_collate(batch)


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
            label += 3**d * action[args.action_dim - d - 1]
        return label

    def strip_sd(state_dict, prefix):
        """
        strip prefix from state dictionary
        """
        return {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    val_csv = pd.read_csv(args.val_csv)
    if args.episode is None:
        val_set = torch.utils.data.ConcatDataset(
            [
                ImitationDatasetFramestackMulti(
                    args.val_csv, args, i, args.data_folder, train=False
                )
                for i in range(min(10, len(val_csv)))
            ]
        )
    else:
        val_set = ImitationDatasetFramestackMulti(
            args.val_csv, args, args.episode, args.data_folder, train=False
        )

    val_loader = DataLoader(val_set, 1, num_workers=4)  # , collate_fn=collate_fn
    with torch.no_grad():
        # construct model
        v_encoder = make_vision_encoder(args.encoder_dim)
        if args.use_flow:
            t_encoder = make_tactile_flow_encoder(args.encoder_dim)
        else:
            t_encoder = make_tactile_encoder(args.encoder_dim)
            # t_encoder = make_tactile_encoder(args.embed_dim_t)
        # a_encoder = make_audio_encoder(2048)
        a_encoder = make_audio_encoder(args.num_stack * args.encoder_dim)
        v_encoder.eval()
        t_encoder.eval()
        a_encoder.eval()

        actor = Actor(v_encoder, t_encoder, a_encoder, args)
        # get pretrained parameters
        state_dict = strip_sd(torch.load(args.pretrained)["state_dict"], "actor.")
        # print("Model's state_dict:")
        # for param_tensor in state_dict:
        #     print(param_tensor, "\t", state_dict[param_tensor].size())
        actor.load_state_dict(state_dict)
        actor.cuda()
        actor.eval()

    cnt = 0  # -args.start_idx
    cor = np.zeros(args.action_dim)
    wrong = np.zeros(args.action_dim)
    total_wrong = 0
    total_cor = 0

    pred_actions = []
    real_actions = []
    pred_labels = []
    real_labels = []
    if args.pouring:
        pred_label_cnts = np.zeros(3**args.action_dim)
        real_label_cnts = np.zeros(3**args.action_dim)
        label_correct = np.zeros(3**args.action_dim)
        label_total = np.zeros(3**args.action_dim)
    else:
        pred_label_cnts = np.zeros(6)
        real_label_cnts = np.zeros(6)
        label_correct = np.zeros(6)
        label_total = np.zeros(6)

    debug_info = True

    model_dir = "/".join(args.pretrained.split("/")[:-1])

    if args.save_video:
        video_saver = MakeVideo(
            dir=model_dir,
            framestack=args.num_stack,
            name=args.exp_name,
            args=args,
            length=len(val_loader),
        )
        video_saver.initialize_sep()

    for batch in tqdm(val_loader):
        # if cnt < 150:
        #     cnt += 1
        #     continue
        # v_gripper_inp, v_fixed_inp, _, _, keyboard = batch
        v_input, t_input, log_spec, keyboard, audio_clip = batch
        if args.save_video:
            # spec_nomel = torch.fft.rfft(audio_clip.type(torch.FloatTensor))
            # # print(spec_nomel.shape)
            # # print(audio_clip.shape)

            # # plt.figure(0)
            # fig_audio, arrs_audio = plt.subplots(2, figsize=(6, 3))
            # # print(x)
            # ## plot mel spec
            # arrs_audio[0].imshow(log_spec[0][0])
            # # arrs_audio[1].imshow(log_spec[0][1])
            # ## plot rfft
            # x = torch.fft.rfftfreq(len(audio_clip[0][0]), 1 / 44100)
            # # # arrs_audio[0].plot(x[:10000], np.abs(spec_nomel[0][0])[:10000])
            # arrs_audio[1].plot(x[:10000], np.abs(spec_nomel[0][0])[:10000])
            # # arrs_audio[0].set_ylim([0,2])
            # # arrs_audio[1].set_ylim([0,2])

            # # plt.imshow(spec_nomel[0][0])
            # video_saver.save_obs(fig_audio, item='audio', step=cnt)
            # # plt.show()
            # # plt.pause(.001)
            # # plt.draw()
            # plt.close(fig_audio)
            video_saver.save_obs([log_spec, audio_clip], item="audio", step=cnt)

        v_input = Variable(v_input).cuda()
        t_input = Variable(t_input).cuda()
        a_input = Variable(log_spec).cuda()

        v_input.squeeze_(0)
        t_input.squeeze_(0)
        s_v = v_input.shape
        s_t = t_input.shape

        ## debugging
        # print(s_v)
        # for i in range(s_v[0]):
        #     img = v_input[i].cpu().permute(1, 2, 0).numpy()
        #     img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        #     cv2.imshow('fixed ' + str(i), img)
        #     cv2.waitKey(200)

        keyboard = keyboard.numpy()
        action_logits, weights = actor(v_input, t_input, a_input, True)
        action_logits = action_logits.detach().cpu().numpy()
        if weights is not None:
            weights = weights.detach().cpu().numpy()

        if args.loss_type == "cce":
            pred_label = np.argmax(action_logits)
            if args.pouring:
                pred_label = pouring_action_map[pred_label]
                gt_label = 0
                pred_temp = pred_label
                pred_action = np.zeros(args.action_dim)
                for d in range(args.action_dim):
                    gt_label += 3**d * keyboard[0][args.action_dim - d - 1]
                    pred_action[args.action_dim - d - 1] = pred_temp % 3
                    pred_temp //= 3
            else:
                pred_action = np.array(key_insertion_action_map[pred_label])
                tmp = np.zeros(keyboard.shape[0])
                # print(tmp)
                tmp[keyboard[:, 0] == 0] = 0
                tmp[keyboard[:, 0] == 2] = 1
                tmp[keyboard[:, 1] == 0] = 2
                tmp[keyboard[:, 1] == 2] = 3
                tmp[keyboard[:, 2] == 0] = 4
                tmp[keyboard[:, 2] == 2] = 5
                gt_label = int(tmp[0])
            # print(f"keyboard {keyboard}, label {gt_label}")
            # print(f"pred_action {pred_action}, label {pred_label}")
        elif args.loss_type == "mse":
            pred_action = pred_action.reshape(-1)  # * np.array((.003, .003, .0015))
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
        for i in range(pred_action.shape[0]):
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
        for i in range(pred_action.shape[0]):
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
            video_saver.save_obs(v_input, "v", pred_action, keyboard, cnt - 1)
            video_saver.save_obs(t_input, "t")

            ## time history
            # fig, axs = plt.subplots(args.action_dim + 1, 1, figsize=(5, 5), sharex='col')
            # titles = ['class', 'x', 'y', 'z']
            # if args.action_dim == 4:
            #     titles.append('dz')
            if args.pouring:
                titles = ["class", "x", "dy"]
            else:
                titles = ["class", "x", "y", "z"]
            for i in range(len(titles)):
                if i < 1:
                    if gt_label == pred_label:
                        mycolor = "b"
                    else:
                        mycolor = "r"
                    video_saver.axs_seq[i].set_title(
                        f"(step {cnt-1}) gt: {gt_label}, pred: {pred_label}",
                        color=mycolor,
                        fontsize=8,
                    )
                    video_saver.axs_seq[i].plot(real_labels, "b+", label="real")
                    video_saver.axs_seq[i].plot(pred_labels, "r.", label="predict")
                else:
                    if keyboard[i - 1] == pred_action[i - 1]:
                        mycolor = "b"
                    else:
                        mycolor = "r"
                    video_saver.axs_seq[i].set_title(
                        f"({titles[i]}) gt: {keyboard[i-1]}, pred: {pred_action[i-1]}",
                        color=mycolor,
                        fontsize=8,
                    )
                    # _real = np.array(real_actions)
                    # _pred = np.array(pred_actions)
                    video_saver.axs_seq[i].plot(
                        cnt - 1, keyboard[i - 1], "b+", label="real"
                    )
                    video_saver.axs_seq[i].plot(
                        cnt - 1, pred_action[i - 1], "r.", label="predict"
                    )
                # axs[i].legend()
            # fig.savefig(os.path.join(model_dir, f"{args.exp_name}_label_t.png"))

            video_saver.save_obs(video_saver.fig_seq, "seq", step=cnt - 1)

            # ## histogram
            # fig, axs = plt.subplots(figsize=(5, 5))
            # axs.set_ylim(0, 300)
            # axs.plot([0, 3 ** args.action_dim], [0, 0], 'k', linewidth=.5)
            # nonzero_labels = list(np.where(pred_label_cnts + real_label_cnts > 0)[0])
            # # print(nonzero_labels)
            # for label in nonzero_labels:
            #     if pred_label_cnts[label] > 0 or real_label_cnts[label] > 0:
            #         # print(f"plot label {label}, real cnt {real_label_cnts[label]}, pred cnt {pred_label_cnts[label]}")
            #         axs.plot([label, label], [0, real_label_cnts[label]], 'b')
            #         axs.plot([label+.5, label+.5], [0, pred_label_cnts[label]], 'r')
            #         max_label = max(real_label_cnts[label], pred_label_cnts[label])
            #         axs.text(x=label, y=-5, s=str(label), fontsize=7)
            #         action = label_to_action(label)
            #         axs.text(x=label, y=max_label, s=f"{action}", rotation=30, fontsize=7)
            # axs.set_title(f"gt {keyboard}, pred {pred_action}, step {cnt-1}")
            # axs.set_ylabel('count of occurrence')
            # axs.set_xlabel('class')

            # video_saver.save_obs(fig, 'hist', step=cnt-1)

            # plt.close(fig)

            # Confusion matrix
            if weights is not None:  # weights.all() != None:
                weights = weights[0]
                modalities = args.ablation.split("_")
                use_vision = "v" in modalities
                use_tactile = "t" in modalities
                use_audio = "a" in modalities
                used_input = []
                output = []
                if use_vision:
                    used_input.append("v_in")
                    output.append("v_out")
                if use_tactile:
                    used_input.append("t_in")
                    output.append("t_out")
                if use_audio:
                    used_input.append("a_in")
                    output.append("a_out")
                df_cm = pd.DataFrame(weights, index=output, columns=used_input)
                plt.figure(figsize=(5, 4))
                sn.set(font_scale=1.4)
                fig_ = sn.heatmap(
                    df_cm, annot=True, cmap="YlGnBu", annot_kws={"fontsize": 20}
                ).get_figure()

                video_saver.save_obs(fig_, "confusion", step=cnt - 1)

                plt.close(fig_)

    ## accuracy summary
    acc = cor / (cor + wrong)
    print(f"each direction acc: {acc}")
    acc = total_cor / (total_wrong + total_cor)
    print(f"class acc: {label_correct / label_total}")
    print(f"EM = {np.sum(label_correct) / np.sum(label_total)}")
    predict = np.asarray(pred_actions)
    real = np.asarray(real_actions)

    ## plot prediction history
    if args.pouring:
        titles = [f"acc: {acc}", "x", "dy"]
    else:
        titles = [f"acc: {acc}", "x", "y", "z"]
    fig, axs = plt.subplots(len(titles), 1, figsize=(15, 10), sharex="col")
    for i in range(len(titles)):
        if i < 1:
            axs[i].plot(real_labels, "b+", label="real")
            axs[i].plot(pred_labels, "r.", label="predict")
        else:
            axs[i].plot(real[:, i - 1], "b+", label="real")
            axs[i].plot(predict[:, i - 1], "r.", label="predict")
        axs[i].title.set_text(titles[i])
        axs[i].legend()
    print(args.pretrained.split("/")[:-1] + [f"{args.exp_name}.png"])
    fig.savefig(os.path.join(model_dir, f"{args.exp_name}_seq.png"))
    plt.close(fig)

    ## trying better visualization
    # fig, axs = plt.subplots()
    # axs.plot([0, 3 ** args.action_dim], [0, 0], 'k', linewidth=.5)
    # for label in range(3 ** args.action_dim):
    #     if pred_label_cnts[label] > 0 or real_label_cnts[label] > 0:
    #         print(f"plot label {label}, real cnt {real_label_cnts[label]}, pred cnt {pred_label_cnts[label]}")
    #         axs.plot([label, label], [0, real_label_cnts[label]], 'b')
    #         axs.plot([label+.5, label+.5], [0, pred_label_cnts[label]], 'r')
    #         max_label = max(real_label_cnts[label], pred_label_cnts[label])
    #         axs.text(x=label, y=-5, s=str(label), fontsize=7)
    #         action = label_to_action(label)
    #         axs.text(x=label, y=max_label, s=f"{action}", rotation=30, fontsize=7)
    # axs.set_ylabel('count of occurrence')
    # axs.set_xlabel('class')
    # axs.set_title(f'acc: {acc}')
    # fig.savefig(os.path.join(model_dir, f"{args.exp_name}_labeldist.png"))

    # fig.show()


if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    # p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
    p.add(
        "-c",
        "--config",
        is_config_file=True,
        default="conf/imi/imi_learn_ablation.yaml",
    )
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=100)
    p.add("--num_workers", default=4, type=int)
    p.add("--start_idx", default=0, type=int)
    # model
    # p.add("--embed_dim", required=True, type=int)
    p.add("--pretrained", required=True)
    p.add("--freeze_till", required=True, type=int)
    p.add("--episode", default=None, type=int)
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--encoder_dim", required=True, type=int)
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
    p.add("--pouring", default=False, type=bool)
    p.add("--cam_to_use", default="fixed")
    # data
    p.add("--crop_percent", default=0.1, type=float)
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="../data_0408/test_recordings/")
    p.add("--num_camera", required=True, type=int)
    p.add("--total_episode", required=True, type=int)
    p.add("--ablation", required=True)
    p.add("--use_layernorm", default=False, type=bool)
    p.add("--exp_name", default=None)
    p.add("--save_video", default=False, action="store_true")
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--norm_freq", default=False, action="store_true")
    p.add("--pool_a_t", default=False, action="store_true")
    p.add("--no_res_con", default=False, action="store_true")

    args = p.parse_args()
    baselineValidate(args)
