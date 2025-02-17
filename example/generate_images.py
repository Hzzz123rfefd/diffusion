import sys
import os
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
from src.dataset import *
from src import datasets,models
from src.utils import *
from torchvision.utils import save_image

def main(args):
    config = load_config(args.model_config_path)

    """ get net struction"""
    model = models[config["model_type"]](**config["model"])
    model.load_pretrained(config["logging"]["save_dir"])

    if args.label == -1:
        image = model.sample()
    else :
        image = model.sample(label = args.label)
    save_image(image[0:1,:,:,:], args.save_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/cddpm.yml")
    parser.add_argument("--save_image_path", type=str, default = "assets/output.png")
    parser.add_argument("--label", type = int, default = -1)
    args = parser.parse_args()
    main(args)