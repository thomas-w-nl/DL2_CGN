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
    # imagenet = torchvision.datasets.ImageNet("/storage/twiggers/",
    #                                          transform=torchvision.transforms.Resize((256, 256)))

    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()])

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

    # Setup GAN network
    model_d = models.resnet18(pretrained=False)
    model_d.fc = nn.Linear(512 * 1, 2)

    model_g = U2NETP(3, 3)

    model_d.to(device)
    model_g.to(device)

    toggle_grad(model_d, True)
    toggle_grad(model_g, True)

    # Setup training utilities
    discriminator_optimizer = torch.optim.Adam(model_d.parameters(), lr=args.lr * 0.1)
    generator_optimizer = torch.optim.Adam(model_g.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=args.reduce_lr_on, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    fake_dataloder1 = get_fake_dataloader(args)
    fake_dataloder2 = get_fake_dataloader(args)
    real_loader = get_real_dataloader(args, len(fake_dataloder1.dataset))
    # real_loader = get_fake_dataloader(args)

    # logging
    run = wandb.init(project="dl2-refinement", entity="thomas-w",
                     config=vars(args),
                     # group=args.model,
                     notes=f"{args.notes}", reinit=True)

    print("Pretraining Unet")
    pretrain_optimizer = torch.optim.Adam(model_g.parameters(), lr = 1e-2)
    for i in range(10):

        criterion_pretrain = nn.MSELoss()
        total = 0
        for i, img in enumerate(fake_dataloder1):
            img = img.to(device)
            # Generator gradient step
            pretrain_optimizer.zero_grad()
            images_fake = model_g(img)
            loss = criterion_pretrain(images_fake, img)
            loss.backward()
            pretrain_optimizer.step()
            total += loss.item()

            # Pretraining is done quite quickly
            if i == 500:
                break
        loss = total / len(fake_dataloder1)
        print("Loss", loss)
        if i == 6:
            pretrain_optimizer = torch.optim.Adam(model_g.parameters(), lr=1e-3)

        # im = torchvision.utils.make_grid(torch.cat((images_fake.cpu(), img.cpu())), ).clip(0, 1).permute((1, 2, 0))
        # plt.imshow(im)
        # plt.show()

    torch.save(model_g.state_dict(), "u2net_pretraining.pt")

    print("Loaded pretrained unet")
    model_g.load_state_dict(torch.load("u2net_pretraining.pt"))



    # generate data
    for epoch in trange(args.epochs):
        loss_total_discriminator = []
        loss_total_generator = []
        total_accuracy = []
        last_total_accuracy = .5

        for fake_1, fake_2, (real, _) in zip(fake_dataloder1, fake_dataloder2, real_loader):
            x_gen1 = fake_1.to(device)
            x_gen2 = fake_2.to(device)
            real = real.to(device)

            # Create fake images
            with torch.no_grad():
                images_fake = model_g(x_gen1)
            labels_fake = torch.zeros((images_fake.shape[0],), device=device).long()

            # append fake and real
            combined_real_fake = torch.cat([images_fake, real])
            combined_labels = torch.cat([labels_fake, labels_fake + 1])

            # Discriminator gradient step
            discriminator_optimizer.zero_grad()
            pred = model_d(combined_real_fake)
            loss = criterion(pred, combined_labels)
            loss.backward()


            loss_total_discriminator.append(loss.cpu().item())
            accuracy = torch.sum(pred.argmax(1) == combined_labels).detach().cpu() / len(combined_labels)
            total_accuracy.append(accuracy)

            if accuracy < .8:
                discriminator_optimizer.step()

            # Generator gradient step
            generator_optimizer.zero_grad()

            images_fake = model_g(x_gen2)
            labels_fake = torch.zeros((images_fake.shape[0],), device=device).long()

            toggle_grad(model_d, False)
            pred = model_d(images_fake)
            toggle_grad(model_d, True)

            loss = criterion(pred, labels_fake)
            loss.backward()
            generator_optimizer.step()

            loss_total_generator.append(loss.cpu().item())

        # lr = scheduler.get_last_lr()
        lr = discriminator_optimizer.param_groups[0]['lr']
        scheduler.step()

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
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--reduce_lr_on', nargs="+", type=int, default=[],
                        help="Milestones for MultiStepLR")
    parser.add_argument('--notes', type=str, default="",
                        help='Notes for wandb')
    parser.add_argument('--dataset', type=str, default="imagenet/data/refinement1/",
                        help='Path to the dataset')
    parser.add_argument('--batch_sz', type=int, default=16,
                        help='Batch size, default 16')

    args = parser.parse_args()

    main(args)
