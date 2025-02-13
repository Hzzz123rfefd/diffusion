from src.model import Diffusion, DDPM
from src.dataset import DatasetForImageGeneration


datasets = {
    "image_generation": DatasetForImageGeneration,
}

models = {
    "diffusion":Diffusion,
    "ddpm":DDPM
}
