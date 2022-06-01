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

import PIL
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

    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()])

    try:
        # raise RuntimeError("I hate imagenet")
        path = "/storage/twiggers/"
        if os.uname()[1] == "robolabws7":
            path = "/storage3/twiggers/"


        imagenet = torchvision.datasets.ImageNet(path,
                                                 transform=tf)
        print("Using imagenet!")
    except RuntimeError as e:
        print(f"[NOTE] Imagenet not found, using cifar10 ({e})")
        imagenet = torchvision.datasets.CIFAR10("imagenet/data/cifar/",
                                                transform=tf, download=True)





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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fake_dataloder1 = get_fake_dataloader(args)
    fake_dataloder2 = get_fake_dataloader(args)
    real_loader = get_real_dataloader(args, len(fake_dataloder1.dataset))

    if not args.pretrained:
        print("NOTE: Not using pretrained networks")

    # Setup GAN network
    model_d = models.resnet18(pretrained=args.pretrained)
    model_d.fc = nn.Linear(512 * 1, 1)

    model_g = U2NETP(3, 3)

    model_d.to(device)
    model_g.to(device)

    toggle_grad(model_d, True)
    toggle_grad(model_g, True)


    # Setup training utilities
    # discriminator_optimizer = torch.optim.Adam(model_d.parameters(), lr=args.lr * 0.1, betas=(.5, 0.999))
    discriminator_optimizer = torch.optim.SGD(model_d.parameters(), lr=args.lr * .2)
    generator_optimizer = torch.optim.Adam(model_g.parameters(), lr=args.lr, betas=(.5, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=args.reduce_lr_on, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # logging
    run = wandb.init(project="dl2-refinement", entity="thomas-w",
                     config=vars(args),
                     # group=args.model,
                     notes=f"{args.notes}", reinit=True)

    criterion_mse = nn.L1Loss() # L1 loss from pix2pix paper

    # print("Pretraining Unet")
    # pretrain_optimizer = torch.optim.Adam(model_g.parameters(), lr = 1e-2)
    # for i in range(10):
    #
    #
    #     total = 0
    #     for i, img in enumerate(fake_dataloder1):
    #     # for i, (img, label) in enumerate(real_loader):
    #         img = img.to(device)
    #         # Generator gradient step
    #         pretrain_optimizer.zero_grad()
    #         images_fake = model_g(img)
    #         loss = criterion_mse(images_fake, img)
    #         loss.backward()
    #         pretrain_optimizer.step()
    #         total += loss.item()
    #
    #         # Pretraining is done quite quickly
    #         if i == 500:
    #             break
    #     loss = total / len(fake_dataloder1)
    #     print("Loss", loss)
    #     if i == 6:
    #         pretrain_optimizer = torch.optim.Adam(model_g.parameters(), lr=1e-3)

        # im = torchvision.utils.make_grid(torch.cat((images_fake.cpu(), img.cpu())), ).clip(0, 1).permute((1, 2, 0))
        # plt.imshow(im)
        # plt.show()

    # torch.save(model_g.state_dict(), "u2net_pretraining.pt")
    #
    if args.pretrained:
        print("Loaded pretrained unet")
        model_g.load_state_dict(torch.load("u2net_pretraining.pt"))

    # generate data
    for epoch in trange(args.epochs):
        loss_total_discriminator = []
        loss_total_generator = []
        total_accuracy = []
        last_total_accuracy = 0

        for (fake_1, _), (fake_2, _), (real, _) in zip(fake_dataloder1, fake_dataloder2, real_loader):
            toggle_grad(model_d, True)
            x_gen1 = fake_1.to(device)
            x_gen2 = fake_2.to(device)
            real = real.to(device)

            # Create fake images
            with torch.no_grad():
                images_fake = model_g(x_gen1)
            labels_fake = torch.zeros((images_fake.shape[0],), device=device)
            # labels_real = labels_fake + .9  # 0.9 is chosen as label smoothing, to reduce discriminator confidence
            labels_real = labels_fake + 1  # 0.9 breaks accuracy metric?


            ############################
            # Discriminator gradient step
            ############################
            discriminator_optimizer.zero_grad()
            pred = model_d(real).squeeze(1)
            loss_real = criterion(pred, labels_real)
            loss_real.backward()

            # # only optimize discriminator if it is not too good
            # if last_total_accuracy < .8:
            #     discriminator_optimizer.step()

            loss_total_discriminator.append(loss_real.cpu().item())
            accuracy_real = torch.sum((torch.sigmoid(pred) >= .5)).detach().cpu() / len(labels_real)
            total_accuracy.append(accuracy_real)

            pred = model_d(images_fake).squeeze(1)
            loss_fake = criterion(pred, labels_fake)
            loss_fake.backward()


            # only optimize discriminator if it is not too good
            # if last_total_accuracy < .8:
            discriminator_optimizer.step()


            loss_total_discriminator.append(loss_fake.cpu().item())
            accuracy_fake = torch.sum((torch.sigmoid(pred) < .5)).detach().cpu() / len(labels_fake)
            total_accuracy.append(accuracy_fake)


            ############################
            # Generator gradient step
            ############################
            generator_optimizer.zero_grad()

            images_fake = model_g(x_gen2)
            labels_fake = torch.ones((images_fake.shape[0],), device=device) # Flipped labels to maximize log D

            toggle_grad(model_d, False)
            pred = model_d(images_fake).squeeze(1)
            #toggle_grad(model_d, True)
            loss = criterion(pred, labels_fake) + criterion_mse(x_gen2, images_fake)
            loss.backward()
            generator_optimizer.step()

            loss_total_generator.append(loss.cpu().item())

        # lr = scheduler.get_last_lr()
        lr = discriminator_optimizer.param_groups[0]['lr']
        scheduler.step()

        print("Accuracies")
        print("real", accuracy_real)
        print("fake", accuracy_fake)

        loss_generator = np.mean(loss_total_generator)
        loss_discriminator = np.mean(loss_total_discriminator)
        total_accuracy = np.mean(total_accuracy)
        last_total_accuracy = total_accuracy
        # print("avg loss_generator:    ", loss_generator)
        # print("avg loss_discriminator:", loss_discriminator)

        image = wandb.Image(torchvision.utils.make_grid(images_fake).clip(0, 1))
        # print("image.shape", image.shape)
        #
        log = {
            "epoch": epoch,
            "loss_generator": loss_generator,
            "loss_discriminator": loss_discriminator,
            "discriminator_accuracy": total_accuracy,
            "lr": lr,
            "images": image,

        }
        wandb.log(log)

    # filename = f"trained_{datetime.now().strftime('%d-%m-%Y_%H.%M.%S')}.pt"
    torch.save(model_d, "trained_last_discriminator.pt")
    torch.save(model_g, "trained_last_generator.pt")
    # torch.save(model, filename)
    # print("Saved weights as", filename)
    print("Saved model")
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, required=True,
    #                     choices=['random', 'best_classes', 'fixed_classes'],
    #                     help='Choose between random sampling, sampling from the best ' +
    #                          'classes or the classes passed to args.classes')
    parser.add_argument('--epochs', type=int, default=100,
                        help='How many epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, # From DCGAN paper
                        help='Learning rate')
    parser.add_argument('--reduce_lr_on', nargs="+", type=int, default=[],
                        help="Milestones for MultiStepLR")
    parser.add_argument('--notes', type=str, default="",
                        help='Notes for wandb')
    parser.add_argument('--dataset', type=str, default="imagenet/data/refinement1/",
                        help='Path to the dataset')
    parser.add_argument('--batch_sz', type=int, default=16,
                        help='Batch size, default 16')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights generator and discriminator')

    args = parser.parse_args()

    main(args)
