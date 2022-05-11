'''
Generate a dataset with the CGN.
The labels are stored in a csv
'''

import warnings
import pathlib
from os.path import join
from datetime import datetime
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import repackage

repackage.up()

from imagenet.dataloader import RefinementDataset, UnNormalize
from imagenet.models import CGN, U2NET, DiceLoss
from utils import toggle_grad


def save_image(im, path):
    torchvision.utils.save_image(im.detach().cpu(), path, normalize=True)


# Lists of best or most interesting shape/texture/background classes
# (Yes, I know all imagenet classes very well by now)
MASKS = [9, 18, 22, 35, 56, 63, 96, 97, 119, 207, 225, 260, 275, 323, 330, 350, 370, 403, 411,
         414, 427, 438, 439, 441, 460, 484, 493, 518, 532, 540, 550, 559, 561, 570, 604, 647,
         688, 713, 724, 749, 751, 756, 759, 779, 780, 802, 814, 833, 841, 849, 850, 859, 869,
         872, 873, 874, 876, 880, 881, 883, 894, 897, 898, 900, 907, 930, 933, 945, 947, 949,
         950, 953, 963, 966, 967, 980]
FOREGROUND = [12, 15, 18, 25, 54, 66, 72, 130, 145, 207, 251, 267, 271, 275, 293, 323, 385,
              388, 407, 409, 427, 438, 439, 441, 454, 461, 468, 482, 483, 486, 490, 492, 509,
              530, 555, 607, 608, 629, 649, 652, 681, 688, 719, 720, 728, 737, 741, 751, 756,
              779, 800, 810, 850, 852, 854, 869, 881, 907, 911, 930, 936, 937, 938, 941, 949,
              950, 951, 954, 957, 959, 963, 966, 985, 987, 992]
BACKGROUNDS = [7, 9, 20, 30, 35, 46, 50, 65, 72, 93, 96, 97, 119, 133, 147, 337, 350, 353, 354,
               383, 429, 460, 693, 801, 888, 947, 949, 952, 953, 955, 958, 970, 972, 973, 974,
               977, 979, 998]


def sample_classes(mode, classes=None):
    if mode == 'random':
        return np.random.randint(0, 1000, 3).tolist()

    elif mode == 'best_classes':
        return [np.random.choice(MASKS),
                np.random.choice(FOREGROUND),
                np.random.choice(BACKGROUNDS)]

    elif mode == 'fixed_classes':
        return [int(c) for c in classes]

    else:
        assert ValueError("Unknown sample mode {mode}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    u2net = torch.load(args.weights, map_location=device)

    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss

    dataset = RefinementDataset("imagenet/data/small1/")

    print("Dataset length:", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0)

    loss_total = []

    with torch.no_grad():
        for data in dataloader:
            x_gt = data["gt"].to(device)
            mask = data["mask"].to(device)
            foreground = data["fg"].to(device)
            background = data["bg"].to(device)

            x_gen = mask * foreground + (1 - mask) * background

            input = torch.hstack((mask * foreground, (1 - mask) * background))

            input = input.detach()
            x_gt = x_gt.detach()

            x_gen_refined = u2net(input)


            ### visualisation

            a = x_gt
            b = x_gen
            c = x_gen_refined

            # a = unnormalize(a)
            # b = unnormalize(b)
            # c = unnormalize(c)

            a = a.cpu()[0].permute(1, 2, 0).numpy()
            b = b.cpu()[0].permute(1, 2, 0).numpy()
            c = c.cpu()[0].permute(1, 2, 0).numpy()

            plt.axis('off')
            fig, axs = plt.subplots(2, 2)
            map(lambda x: x.axis('off'), axs.ravel())

            axs[0, 0].imshow(a)
            axs[0, 0].set_title("x_gt")

            axs[1, 0].imshow(b)
            axs[1, 0].set_title("x_gen")

            axs[1, 1].imshow(c)
            axs[1, 1].set_title("x_gen_refined")
            plt.show()

            loss = criterion(x_gen_refined, x_gt)

            loss_total.append(loss.cpu().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, required=True,
    #                     choices=['random', 'best_classes', 'fixed_classes'],
    #                     help='Choose between random sampling, sampling from the best ' +
    #                          'classes or the classes passed to args.classes')
    parser.add_argument('--weights', type=str, required=True,
                        help='which weights to use')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for the sampling the noise')

    args = parser.parse_args()
    # if args.mode != 'fixed_classes' and [0, 0, 0] != args.classes:
    #     warnings.warn(f"You supply classes, but they won't be used for mode = {args.mode}")
    main(args)
