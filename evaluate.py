import argparse
import torch
from imagenet.models import RefineNetShallow, DiceLoss
from imagenet.dataloader import RefinementDataset
from imagenet.blending_utils import poisson_blending, copy_paste_blending
from utils import toggle_grad
from torch import nn
import numpy as np
import cv2
import time


class RefinementWrapper():
    def __init__(self):
        self.model = RefineNetShallow()

    def predict(self, foreground, background, mask):
        input = torch.hstack((mask * foreground, (1 - mask) * background))
        input = input.detach()

        return self.model(input)


class CopyPasteWrapper:
    def predict(self, foreground, background, mask):
        return copy_paste_blending(foreground, background, mask)


class PoissonWrapper:
    def predict(self, foreground, background, mask):
        return poisson_blending(foreground, background, mask)


def load_refinemnet_model(args):
    model = RefinementWrapper()
    model.to(args.device)
    toggle_grad(model, False)


def load_test_set(args):
    dataset = RefinementDataset(args.dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True)

    return loader


def load_criterion(args):
    criterions = {'ce': nn.CrossEntropyLoss(),
                  'dice': DiceLoss,
                  'mse': nn.MSELoss()}
    if args.criterion not in criterions:
        raise ValueError("{args.criterion} is an unknown criterion")

    return criterions[args.criterion]


def save_scores(scores, args):
    date_str = time.strftime("%Y%m%d_%H%M%S")
    np.save(f'scores_model_{args.model_type}_criterion_{args.criterion}_{date_str}.npy', scores)


def main(args):
    if args.model_type == "refinement":
        model = load_refinemnet_model(args)
    elif args.model_type == "poisson":
        model = PoissonWrapper()
    elif args.model_type == "copy_paste":
        model = CopyPasteWrapper()
    else:
        raise ValueError(f"Model {args.model_type} not known.")

    dataloader = load_test_set(args)
    criterion = load_criterion(args)
    scores = []
    for data in dataloader:
        x_gt = data["gt"].to(device)
        mask = data["mask"].to(device)
        foreground = data["fg"].to(device)
        background = data["bg"].to(device)

        model_pred = model.predict(foreground, background, mask)
        cv2.imshow(model_pred)
        cv2.imshow(x_gt)
        cv2.waitKey(0)

        scores.append(criterion(model_pred, x_gt))

    save_scores(scores, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='How many epochs to train for')
    parser.add_argument('--model_type', type=str, required=True, choices=('copy_paste', 'poisson', 'refinement'), help='Model type to evaluate')
    parser.add_argument('--dataset', type=str, default="imagenet/data/refinement1/", help='Path to the dataset')
    parser.add_argument('--model_path', type=str, help="Path to model. Only required if model_type=refinement.")
    parser.add_argument('--batch_sz', type=int, default=8, help='Batch size, default 8')
    parser.add_argument('--criterion', type=str, required=True, choices=('ce', 'mse', 'dice'), help='Criterion metric to evaluate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.model_type == 'refinement' else 'cpu')
    args.device = device

    main(args)
