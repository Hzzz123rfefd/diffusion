from src.model import Diffusion
from src.dataset import DatasetForImageGeneration


datasets = {
    "image_generation": DatasetForImageGeneration,
}

models = {
    "diffusion":Diffusion,
}
