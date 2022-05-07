import os

import torch
import torch.nn.functional as F
import argparse

import repackage
from PIL import Image
from torchvision import transforms
import tqdm

repackage.up()

from imagenet.models import CGN


def generate_images(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cgn = CGN(
        batch_sz=1,
        truncation=args.truncation,
        pretrained=False,
    )

    dataset_path = args.dataset_path + "/"

    os.makedirs(dataset_path, exist_ok=True)


    weights = torch.load(args.weights_path, map_location="cpu")
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    cgn.load_state_dict(weights)
    cgn.eval().to(device)

    data_dict = {"bg": None, "fg": None, "m": None, "gt": None, "labels": None}
    with torch.no_grad():
        for i in tqdm.trange(args.n_data):

            y_vec = torch.randint(0, 1000, (1,)).to(torch.int64)
            y_vec = F.one_hot(y_vec, 1000).to(torch.float32)

            dev = cgn.get_device()
            u_vec = cgn.get_noise_vec()

            inp = (u_vec.to(dev), y_vec.to(dev), cgn.truncation)
            x_gt, mask, _, foreground, background, _ = cgn(inp=inp)

            x_gt = x_gt.clip(0,1).squeeze(0)
            mask = mask.clip(0,1).squeeze(0)
            foreground = foreground.clip(0,1).squeeze(0)
            background = background.clip(0,1).squeeze(0)

            to_pil = transforms.ToPILImage()

            to_pil(x_gt).save(dataset_path + f"{i}_gt.png")
            to_pil(mask).save(dataset_path + f"{i}_mask.png")
            to_pil(foreground).save(dataset_path + f"{i}_fg.png")
            to_pil(background).save(dataset_path + f"{i}_bg.png")



        #     if i:
        #         data_dict["bg"] = torch.cat((background.cpu(), data_dict["bg"]), dim=0)
        #         data_dict["fg"] = torch.cat((foreground.cpu(), data_dict["fg"]), dim=0)
        #         data_dict["m"] = torch.cat((mask.cpu(), data_dict["m"]), dim=0)
        #         data_dict["gt"] = torch.cat((x_gt.cpu(), data_dict["gt"]), dim=0)
        #         data_dict["labels"] = torch.cat((y_vec, data_dict["labels"]), dim=0)
        #     else:
        #         data_dict["labels"] = y_vec.cpu()
        #         data_dict["bg"] = background.cpu()
        #         data_dict["fg"] = foreground.cpu()
        #         data_dict["m"] = mask.cpu()
        #         data_dict["gt"] = x_gt.cpu()
        #
        # print(data_dict["gt"].shape)

    return


def main(args):

    generate_images(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', type=int, required=True,
                        help='How many datapoints to sample'),
    parser.add_argument('--weights_path', type=str, default="imagenet/weights/cgn.pth",
                        help='Which weights to load for the CGN')
    parser.add_argument('--dataset_path', type=str,  required=True,
                        help='Where to place the dataset')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for the sampling the noise')

    args = parser.parse_args()
    main(args)
