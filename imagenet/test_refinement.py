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

from shared.losses import PerceptualLoss

repackage.up()

from imagenet.blending_utils import poisson_blending
from imagenet.dataloader import RefinementDataset, UnNormalize
from imagenet.models import CGN, U2NET, DiceLoss
from utils import toggle_grad


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.weights, map_location=device)

    criterions = {"Dice": DiceLoss,
                  "Crossentropy": nn.CrossEntropyLoss(),
                  "Perceptual": PerceptualLoss(style_wgts=[4, 4, 4, 4]).to(device)}

    dataset = RefinementDataset("imagenet/data/testset/")

    print("Dataset length:", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0)

    loss_total = {k: 0 for k in criterions}

    """
    Copy-Paste loss:
    Dice 158.76831716299057
    Crossentropy 129.79788411571644
    Perceptual 23046.4071726799

    Model loss:
    Dice 330.1408591270447
    Crossentropy 261.18322888668627
    Perceptual 49224.474759578705

    Poisson loss:
    Dice 192.87030971050262
    Crossentropy 131.06489974749275
    Perceptual 34468.3846783638 
    """

    # # Ground truth
    # with torch.no_grad():
    #     for data in tqdm(dataloader):
    #         x_gt = data["gt"].to(device)
    #         mask = data["mask"].to(device)
    #         foreground = data["fg"].to(device)
    #         background = data["bg"].to(device)
    #
    #         x_gen = mask * foreground + (1 - mask) * background
    #
    #         for name, criterion in criterions.items():
    #             loss_total[name] += criterion(x_gen, x_gt).item()
    #
    # print("Copy-Paste loss:")
    # for name, score in loss_total.items():
    #     print(name, score)
    #
    # # Model
    # with torch.no_grad():
    #     for data in tqdm(dataloader):
    #         x_gt = data["gt"].to(device)
    #         mask = data["mask"].to(device)
    #         foreground = data["fg"].to(device)
    #         background = data["bg"].to(device)
    #
    #         x_gen = mask * foreground + (1 - mask) * background
    #
    #         input = torch.hstack((mask * foreground, (1 - mask) * background))
    #
    #         input = input.detach()
    #         x_gt = x_gt.detach()
    #
    #         x_gen_refined = model(input)
    #
    #         x_gen_refined.detach()
    #
    #         for name, criterion in criterions.items():
    #             loss_total[name] += criterion(x_gen_refined, x_gt).item()
    #
    # print("Model loss:")
    # for name, score in loss_total.items():
    #     print(name, score)

    criterions["Perceptual"].to(torch.device("cpu"))

    # Poisson
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            x_gt = data["gt"]
            mask = data["mask"].squeeze(0).permute(1, 2, 0)
            foreground = (data["fg"].squeeze(0).permute(1, 2, 0) * 255).int()
            background = (data["bg"].squeeze(0).permute(1, 2, 0) * 255).int()


            try:
                x_poisson = poisson_blending(foreground, background, mask)
            except IndexError as e:
                print("Error on image", i)
                continue
            x_poisson = torch.from_numpy(x_poisson).permute(2, 0, 1).unsqueeze(0).contiguous()
            x_poisson = x_poisson.float() / 255.0

            for name, criterion in criterions.items():
                loss_total[name] += criterion(x_poisson, x_gt).item()

    print("Poisson loss:")
    for name, score in loss_total.items():
        print(name, score)


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
