import argparse

import sklearn.metrics
from train_gan import get_real_dataloader
import os
import torch
import numpy as np
import torchvision
from train_classifier import accuracy
from dataloader import ImagenetVanilla
from torch.utils.data import DataLoader


def get_imagenetA(batch_size, data_root_path, workers):
    # dataset
    dataset = ImagenetVanilla(data_root_path, train=False)

    # dataloader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=True, drop_last=True)

    return loader

def eval(args):
    model = torch.load(args.model_path)

    model.eval()

    datasets = args.datasets
    eval_dict = dict()
    for path in datasets:
        data_name = path.split("/")[-1]
        single_dataset_eval = dict()
        _, dataloader, _ = get_imagenetA(args.batch_size, path, args.workers)

        top1, top5, AUPR = [], [], []

        for image_batch, label_batch in dataloader:
            preds = model(image_batch)
            acc1, acc5 = accuracy(preds, label_batch, topk=(1, 5))
            avg_prec_score = sklearn.metrics.average_precision_score(label_batch.numpy(), preds.numpy())

            top1.append(acc1)
            top5.append(acc5)
            AUPR.append(avg_prec_score)
            print(acc1)
            print(acc5)
            print(avg_prec_score)
            break

        single_dataset_eval["top1"] = np.mean(top1)
        single_dataset_eval["top5"] = np.mean(top5)
        single_dataset_eval["AUPR"] = np.mean(AUPR)


        eval_dict[data_name] = single_dataset_eval

        print(args.model_name)
        print("--------------")
        print(f"Top-1 Accuracy: {np.mean(top1)}")
        print(f"Top-5 Accuracy: {np.mean(top5)}")
        print(f"AUPR-score: {np.mean(AUPR)}")

    return eval_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs="+", default="/storage/twiggers/imagenet-a",
                        help="Path(s) to the dataset(s)")
    parser.add_argument("--model_path", type=str, default="model/path",
                        help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")


    args = parser.parse_args()

    eval(args)