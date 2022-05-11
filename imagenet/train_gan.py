'''
Generate a dataset with the CGN.
The labels are stored in a csv
'''
import os
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

from imagenet.dataloader import RefinementDataset
from imagenet.models import CGN, U2NET, DiceLoss, RefineNetShallow, U2NETP
from utils import toggle_grad
from shared.losses import PerceptualLoss
import torchvision.models as models

def get_imagenet_data(n):
    imagenet = torchvision.datasets.ImageNet('.')
    np.random.seed(69)
    imagenet_n = torch.utils.data.Subset(imagenet, np.random.choice(len(imagenet), n, replace=False))

    return imagenet_n

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


def show_images(x_gt, x_gen, x_gen_refined):
    a = x_gt
    b = x_gen
    c = x_gen_refined

    # a = unnormalize(a)
    # b = unnormalize(b)
    # c = unnormalize(c)

    a = a.detach().cpu()[0].permute(1, 2, 0).numpy().clip(0, 1)
    b = b.detach().cpu()[0].permute(1, 2, 0).numpy().clip(0, 1)
    c = c.detach().cpu()[0].permute(1, 2, 0).numpy().clip(0, 1)

    fig, axs = plt.subplots(2, 2)
    for x in axs.ravel():
        x.axis('off')

    axs[0, 0].imshow(a)
    axs[0, 0].set_title("x_gt")

    axs[1, 0].imshow(b)
    axs[1, 0].set_title("x_gen")

    axs[1, 1].imshow(c)
    axs[1, 1].set_title("x_gen_refined")
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup GAN network
    model_d = models.resnet18(pretrained=True)
    model_g = U2NETP(3, 3)

    model_d.to(device)
    model_g.to(device)

    toggle_grad(model_d, True)
    toggle_grad(model_g, True)

    # Setup training utilities
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.reduce_lr_on, gamma=0.1)

    if args.dice_loss:
        criterion = DiceLoss
    else:
        criterion = nn.CrossEntropyLoss()

    criterion = PerceptualLoss(style_wgts=[4, 4, 4, 4])
    criterion.to(device)

    dataset = RefinementDataset(args.dataset)

    print("Dataset length:", len(dataset))

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True)

    # logging
    run = wandb.init(project="dl2-refinement", entity="thomas-w",
                     config=vars(args),
                     # group=args.model,
                     notes=f"{args.notes}", reinit=True)

    # generate data
    for epoch in trange(args.epochs):
        loss_total = []

        for data in trainloader:
            x_gt = data["gt"].to(device)
            mask = data["mask"].to(device)
            foreground = data["fg"].to(device)
            background = data["bg"].to(device)

            x_gen = mask * foreground + (1 - mask) * background

            input = torch.hstack((mask * foreground, (1 - mask) * background))

            input = input.detach()
            x_gt = x_gt.detach()

            optimizer.zero_grad()

            x_gen_ref = model(input)

            loss = criterion(x_gen_ref, x_gt)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.cpu().item())

        if os.getlogin() == "thomas":
            if x_gt is not None:
                show_images(x_gt, x_gen, x_gen_ref)

        # lr = scheduler.get_last_lr()
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        loss = np.mean(loss_total)
        print("avg loss:", loss)
        log = {
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
        }
        wandb.log(log)

    filename = f"trained_{datetime.now().strftime('%d-%m-%Y_%H.%M.%S')}.pt"
    torch.save(model, "trained_last.pt")
    torch.save(model, filename)
    print("Saved weights as", filename)

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, required=True,
    #                     choices=['random', 'best_classes', 'fixed_classes'],
    #                     help='Choose between random sampling, sampling from the best ' +
    #                          'classes or the classes passed to args.classes')
    parser.add_argument('--epochs', type=int, default=30,
                        help='How many epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--reduce_lr_on', nargs="+", type=int, default=[],
                        help="Milestones for MultiStepLR")
    parser.add_argument('--notes', type=str, default="",
                        help='Notes for wandb')
    parser.add_argument('--dataset', type=str, default="imagenet/data/refinement1/",
                        help='Path to the dataset')
    parser.add_argument('--batch_sz', type=int, default=8,
                        help='Batch size, default 32')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for the sampling the noise')

    args = parser.parse_args()
    # if args.mode != 'fixed_classes' and [0, 0, 0] != args.classes:
    #     warnings.warn(f"You supply classes, but they won't be used for mode = {args.mode}")
    main(args)
