import sys
import os
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
from src.dataset import *
from src import datasets,models
from src.utils import *


def main(args):
    config = load_config(args.model_config_path)

    """ get net struction"""
    model = models[config["model_type"]](**config["model"])
    model.load_pretrained(config["logging"]["save_dir"])

    # 生成图片
    image = model.generate_image("cuda")
    image = image.cpu().detach().numpy()
    image = image * 255
    image = image.astype(np.uint8)  #(b,c,h,w)
    # 保存图片
    with open("generate_image.raw", "wb") as file: 
        for h in range(image.shape[2]):
            for w in range(image.shape[3]):
                for c in range(image.shape[1]):
                    file.write(image[0,c,h,w].tobytes())
                file.write(b'\xFF')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/diffusion.yml")
    args = parser.parse_args()
    main(args)