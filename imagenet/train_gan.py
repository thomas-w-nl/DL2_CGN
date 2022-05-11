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

def get_real_dataloader(args, n):
    imagenet = torchvision.datasets.ImageNet('.')
    np.random.seed(69)
    imagenet_n = torch.utils.data.Subset(imagenet, np.random.choice(len(imagenet), n, replace=False))

    print("Real dataset length:", len(imagenet_n))

    loader = torch.utils.data.DataLoader(
        imagenet_n,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True)

    return loader

def get_fake_dataloader(args):
    dataset = RefinementDataset(args.dataset)
    print("Fake dataset length:", len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True)

    return loader

def discriminator_train_batch(model, batch):


def refinement_train_batch():
    pass


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
    discriminator_optimizer = torch.optim.Adam(model_d.parameters(), lr=args.lr)
    generator_optimizer = torch.optim.Adam(model_g.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=args.reduce_lr_on, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    # logging
    run = wandb.init(project="dl2-refinement", entity="thomas-w",
                     config=vars(args),
                     # group=args.model,
                     notes=f"{args.notes}", reinit=True)

    fake_dataloder1 = get_fake_dataloader(args)
    fake_dataloder2 = get_fake_dataloader(args)
    real_loader = get_real_dataloader(args, len(fake_dataloder1.dataset))

    # generate data
    for epoch in trange(args.epochs):
        loss_total = []

        for fake_1, fake_2, real in zip(fake_dataloder1, fake_dataloder2, real_loader):
            x_gen1 = fake_1.to(device)
            x_gen2 = fake_2.to(device)
            real = real.to(device)

            # Create fake images
            with torch.no_grad():
                images_fake = model_g(x_gen1)
            labels_fake = torch.zeros((images_fake.shape[0], 1))

            # append fake and real
            combined_real_fake = np.concatenate([images_fake, real])
            combined_labels = np.concatenate([labels_fake, labels_fake+1])

            # Discriminator gradient step
            discriminator_optimizer.zero_grad()
            pred = model_d(combined_real_fake)

            loss = criterion(pred, combined_labels)
            loss.backward()
            discriminator_optimizer.step()

            # loss_total.append(loss.cpu().item())


            # Generator gradient step
            generator_optimizer.zero_grad()

            images_fake = model_g(x_gen2)
            labels_fake = torch.zeros((images_fake.shape[0], 1))

            with torch.no_grad():
                pred = model_d(images_fake)

            loss = criterion(pred, labels_fake)
            loss.backward()
            generator_optimizer.step()


        if os.getlogin() == "thomas":
            if x_gt is not None:
                show_images(x_gt, x_gen, x_gen_ref)

        # lr = scheduler.get_last_lr()
        lr = discriminator_optimizer.param_groups[0]['lr']
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
