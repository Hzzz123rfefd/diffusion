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

    # 生成图片
    model.eval()
    with torch.no_grad():
        image = model.sample(guide_w = 2)
        image = image.cpu().detach().numpy()
        image = image * 255
        image = image.astype(np.uint8)  #(b,c,h,w)
        image = image.squeeze(0).squeeze(0)
        cv2.imwrite("output.png", image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/ddpm.yml")
    args = parser.parse_args()
    main(args)