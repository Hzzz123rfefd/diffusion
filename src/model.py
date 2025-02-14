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
 
class ModelDiffusionBase(ModelRegression):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 256, 
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(device)
        self.width = width
        self.height = height
        self.channel = channel
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.time_dim = time_dim
        
        for k, v in self.ddpm_schedules(self.beta_start, self.beta_end, self.noise_steps).items():
            self.register_buffer(k, v)
    
    def ddpm_schedules(self, beta1, beta2, T):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")    

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

