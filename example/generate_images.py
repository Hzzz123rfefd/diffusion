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
    # model.load_state_dict(torch.load("data/diffusion_outputs10/model_4.pth"))

    # 生成图片
    model.eval()
    with torch.no_grad():
        image = model.sample(label = 1)
        for i in range(10):
            save_image(image[i:i+1,:,:,:], f"output{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/cddpm.yml")
    args = parser.parse_args()
    main(args)