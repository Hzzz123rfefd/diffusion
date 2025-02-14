from src.model import ModelDDPM, ModelCDDPM
from src.dataset import DatasetForImageGeneration


datasets = {
    "image_generation": DatasetForImageGeneration,
}

models = {
    "ddpm": ModelDDPM,
    "cddpm": ModelCDDPM
}
