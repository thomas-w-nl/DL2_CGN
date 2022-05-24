import argparse

import sklearn.metrics
import torch
import numpy as np
import os
from torchvision import transforms
from train_classifier import accuracy
from torch.utils.data import DataLoader, Dataset
from models.classifier_ensemble import InvariantEnsemble
from dataloader import get_imagenet_dls
import glob
from PIL import Image

def transform_labels(x):
    return torch.tensor(x).to(torch.int64)

class ImagenetA(Dataset) :

    def __init__(self, root_path):
        super(ImagenetA, self).__init__()
        # root = join('.', 'imagenet', 'data')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Transforms
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

        t_list += [transforms.ToTensor(), normalize]
        self.T_ims = transforms.Compose(t_list)

        self.im_paths, self.labels = self.get_data(root_path)

    def set_len(self, n):
        assert n < len(self), "Ratio is too large, not enough CF data available"
        self.im_paths = self.im_paths[:n]
        self.labels = self.labels[:n]

    @staticmethod
    def get_data(p):
        ims, labels = [], []
        subdirs = sorted(glob.glob(p + '/*'))
        for i, sub in enumerate(subdirs):
            im = sorted(glob.glob(sub + '/*'))
            l = np.ones(len(im))*i
            ims.append(im), labels.append(l)
        return np.concatenate(ims), np.concatenate(labels)

    def __getitem__(self, idx):
        ims = Image.open(self.im_paths[idx]).convert('RGB')
        labels = self.labels[idx]
        return {
            'ims': self.T_ims(ims),
            'labels': transform_labels(labels),
        }

    def __len__(self):
        return len(self.im_paths)

def accuracyA(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_imagenetA(batch_size, data_root_path, workers):
    # dataset
    dataset = ImagenetA(data_root_path)

    # dataloader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=True, drop_last=True)

    return loader

def eval(args):
    #model = torch.load(args.model_path)

    model = InvariantEnsemble(arch=args.model_name, pretrained=False, cut=1)

    state_dict = torch.load(args.model_path)['state_dict']

    state_dict = {state_dict_key.replace("module.", ""):state_dict_value for state_dict_key, state_dict_value in state_dict.items()}

    model.load_state_dict(state_dict)

    model.eval()

    eval_dict = dict()

    data_path = args.dataset

    if args.data_name == "imagenet":
        _, dataloader, _ = get_imagenet_dls(False, args.batch_size, args.workers, data_path)
    else:
        dataloader = get_imagenetA(args.batch_size, data_path, args.workers)

    top1_bg, top1_text, top1_shape = [], [], []
    top5_bg, top5_text, top5_shape = [], [], []

    for batch in dataloader:
        image_batch = batch["ims"]
        label_batch = batch["labels"]

        print(label_batch)

        preds = model(image_batch)


        acc1_shape, acc5_shape = accuracyA(preds['shape_preds'], label_batch, topk=(1, 5))
        acc1_text, acc5_text = accuracyA(preds['texture_preds'], label_batch, topk=(1, 5))
        acc1_bg, acc5_bg = accuracyA(preds['bg_preds'], label_batch, topk=(1, 5))

        #avg_prec_score = sklearn.metrics.average_precision_score(label_batch.numpy(), preds.numpy())

        top1_shape.append(acc1_shape)
        top1_text.append(acc1_text)
        top1_bg.append(acc1_bg)

        top5_shape.append(acc5_shape)
        top5_text.append(acc5_text)
        top5_bg.append(acc5_bg)

    eval_dict["top1_bg"] = np.mean(top1_bg)
    eval_dict["top1_text"] = np.mean(top1_text)
    eval_dict["top1_shape"] = np.mean(top1_shape)

    eval_dict["top5_bg"] = np.mean(top5_bg)
    eval_dict["top5_text"] = np.mean(top5_text)
    eval_dict["top5_shape"] = np.mean(top5_shape)


    print("---------------")
    print("Top-1 Accuracy (Shape, Background, Texture): ", eval_dict["top1_shape"], eval_dict["top1_bg"], eval_dict["top1_text"])
    print("Top-5 Accuracy (Shape, Background, Texture): ", eval_dict["top5_shape"], eval_dict["top5_bg"], eval_dict["top5_text"])

    return eval_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="storage/twiggers/imagenet-a",
                        help="Path(s) to the dataset(s)")
    parser.add_argument("--data_name", type=str, default="imagenetA")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Name of model architecture")
    parser.add_argument("--model_path", type=str, default="imagenet/experiments/classifier_2022_05_23_09_39_classifier_INS2_LR_START_0_001_seed_2/model_best.pth",
                        help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of workers for creating dataloader")


    args = parser.parse_args()

    eval(args)