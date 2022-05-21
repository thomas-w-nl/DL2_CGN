import argparse

import sklearn.metrics
from train_gan import get_real_dataloader
import os
import torch
import numpy as np
import torchvision
from train_classifier import accuracy

# TODO Inladen datasets, afhankelijk van implementatie daarvan in de classifier
# TODO Zorgen dat drop last aanstaat ivm makkelijk bereken acc

def get_dataloader(args):

    path = args.path
    n = args.n

    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()])

    try:
        # raise RuntimeError("I hate imagenet")
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

    loader = torch.utils.data.DataLoader(
        imagenet_n,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True)

    return loader

def eval(args):

    if args.name == "":
        # Initialize model class using command line arguments
        # TODO Daadwerkelijke model classes toevoegen
        model = args.ModelClass(*args.model_args)

    # Load Parameters of the trained model
    model.load_state_dict(torch.load(args.model))

    model.eval()

    datasets = args.datasets
    eval_dict = dict()
    for path in datasets:
        data_name = path.split("/")[-1]
        single_dataset_eval = dict()
        dataloader = load_data(path) # TODO Load data functie

        top1, top5, AUPR = [], [], []

        for image_batch, label_batch in dataloader:
            preds = model(image_batch)
            acc1, acc5 = accuracy(preds, label_batch, topk=(1, 5))
            avg_prec_score = sklearn.metrics.average_precision_score(labels.numpy(), preds.numpy())

            top1.append(acc1)
            top5.append(acc5)
            AUPR.append(avg_prec_score)

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

    parser.add_argument("--dataset", nargs="+", default="imagenet/data/refinement",
                        help="Path(s) to the dataset(s)")
    parser.add_argument("--model", type=str, default="model/path",
                        help="Path to the model")
    parser.add_argument("--model_name", type=str, default="baseline_classifier",
                        help="Name of model to be evaluated")
    parser.add_argument("--model_args", nargs="+", default=[],
                        help="Arguments used for loading the model")
    parser.add_argument("--model_class", type=str, default="",
                        help = "Name of the model class")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of images to use in evaluataion")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")


    args = parser.parse_args()

    eval(args)