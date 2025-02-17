import torch
from torch import nn
from einops import rearrange 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch import optim

from src.utils import *
from src.modules import *
from src.base import *


class ModelDDPM(ModelDiffusionBase):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 32, 
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(width, height, channel, time_dim, noise_steps, beta_start, beta_end, device)
        
        self.predict_model = Unet(
            width = self.width,
            height = self.height, 
            in_c = self.channel, 
            out_c = self.channel, 
            time_dim = time_dim
        ).to(self.device)
        
    
    def sample(self, sample_num = 1, guide_w = 0.0):
        self.eval()
        with torch.no_grad():
            x_i = torch.randn(sample_num, self.channel, self.height, self.width).to(self.device)
            print()
            for i in range(self.noise_steps, 0, -1): 
                print(f'sampling timestep {i}',end='\r')
                x_i = x_i.repeat(2,1,1,1)
                t_is = torch.tensor([i]).to(self.device).repeat(2 * sample_num)
                z = torch.randn(sample_num, self.channel, self.height, self.width).to(self.device) if i > 1 else 0
                
                eps = self.predict_model(x_i,t_is)
                eps1 = eps[:sample_num]
                eps2 = eps[sample_num:]
                eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:sample_num]
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            x_i = torch.clamp(x_i, min = 0, max = 1)
        return x_i  #(n, b, h, w)

    def forward(self, inputs):
        x = inputs["image"].to(self.device)
        _ts = torch.randint(1, self.noise_steps+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        predicted_noise = self.predict_model(x_t, _ts)
        output = {
            "predict": predicted_noise,
            "label": noise
        }
        return output
    
class ModelCDDPM(ModelDiffusionBase):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 32, 
        label_dim = 10,
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(width, height, channel, time_dim, noise_steps, beta_start, beta_end, device)
        self.label_dim = label_dim
        self.predict_model = CUnet(
            width = self.width,
            height = self.height, 
            in_c = self.channel, 
            out_c = self.channel, 
            time_dim = self.time_dim, 
            label_dim = self.label_dim
        ).to(self.device)
    
    def sample(self, sample_num = 1, label = None, guide_w = 0.0):
        self.eval()
        with torch.no_grad():
            x_i = torch.randn(sample_num, self.channel, self.height, self.width).to(self.device)
            print()
            for i in range(self.noise_steps, 0, -1): 
                print(f'sampling timestep {i}',end='\r')
                x_i = x_i.repeat(2,1,1,1)
                t_is = torch.tensor([i]).to(self.device).repeat(2 * sample_num)
                label = torch.tensor([label]).to(self.device).repeat(2 * sample_num)
                z = torch.randn(sample_num, self.channel, self.height, self.width).to(self.device) if i > 1 else 0
                
                eps = self.predict_model(x_i,t_is,label)
                eps1 = eps[:sample_num]
                eps2 = eps[sample_num:]
                eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:sample_num]
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            x_i = torch.clamp(x_i, min = 0, max = 1)
        return x_i

    def forward(self, inputs):

        x = inputs["image"].to(self.device)
        label = inputs["label"].to(self.device)
        _ts = torch.randint(1, self.noise_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        predicted_noise = self.predict_model(x_t, _ts, label)
        output = {
            "predict": predicted_noise,
            "label": noise
        }
        return output
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    unet = Unet(in_c=3,out_c=3,time_dim=32).to(device)
    x = torch.randn(batch, 3, 720, 960).to(device)
    t = torch.rand(batch).to(device)
    output_t = unet(x,t)
    print(output_t.shape)

