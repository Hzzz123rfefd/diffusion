import os
import random
import re
import cv2
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from compressai.utils import *

class DatasetForImageGeneration:
    def __init__(
        self,
        train_data_path: str = None,
        test_data_path: str = None,
        valid_data_path: str = None,
        target_height: int = None,
        target_width: int = None,
        data_type:str = "train"
    ):
        self.target_height = target_height
        self.target_width = target_width
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path,split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path,split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path,split = "train")

        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        if len(resized_image.shape) == 2:  
            resized_image = np.expand_dims(image, axis=0)
        return resized_image

    def __getitem__(self, idx):
        output = {}
        image_path = self.dataset[idx]["image_path"]
        label =  self.dataset[idx]["label"]
        output["label"] = torch.tensor(label, dtype = torch.int64)
        output["image"] = torch.from_numpy(self.read_image(image_path)).permute(2,0,1) / 255.0
        return output
    
    def collate_fn(self,batch):
        return  recursive_collate_fn(batch)