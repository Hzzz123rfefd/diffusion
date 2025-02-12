import numpy as np
import torch
from torch import nn
from einops import rearrange 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import os
from torch import optim
import numpy as np
import cv2

from src.utils import *
from src.modules import *

class Diffusion(nn.Module):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 32, 
        noise_steps=500, 
        beta_start=1e-4, 
        beta_end=0.02,
        device = "cuda"
    ):
        super(Diffusion,self).__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.width = width
        self.height = height
        self.channel = channel
        self.time_dim = time_dim
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.unet = Unet(width = self.width,height = self.height, in_c = self.channel, out_c = self.channel, time_dim = time_dim).to(self.device)
        # 1 生成betas
        self.betas = torch.from_numpy(np.linspace(self.beta_start, self.beta_end, self.noise_steps , dtype=np.float32))
        # 2 计算aerfas
        self.aerfas = 1 - self.betas
        # 3 计算aerfa_ba
        self.aerfa_ba = torch.cumprod(self.aerfas, dim=0)
    
    def generate_t_steps(self,batch_size):
        return torch.randint(low=1, high=self.noise_steps+1, size=(batch_size,))
    
    def generate_image(self,device):
        n = 1
        self.betas = self.betas.to(device)
        self.aerfas = self.aerfas.to(device)
        self.aerfa_ba = self.aerfa_ba.to(device)
        self.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channel, self.height, self.width)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                print(i,"...")
                t = (torch.ones(n) * i).float().to(device)
                index = t.cpu().detach().numpy()
                predicted_noise = self.unet(x, t)
                alpha = self.aerfas[index][:, None, None, None]
                alpha_hat = self.aerfa_ba[index][:, None, None, None]
                beta = self.betas[index][:, None, None, None]
                mu = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))* predicted_noise)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # var = torch.sqrt(beta)
                alpha_hat_1 = self.aerfa_ba[index-1][:, None, None, None]
                var = 1 / ((alpha/beta + (1/(1-alpha_hat_1))))
                var = torch.sqrt(var)
                x = mu + var * noise
        x = (x.clamp(0, 1))
        return x
    
    def forward(self, inputs):
        x = inputs["image"].to(self.device)
        self.betas = self.betas.to(self.device)
        self.aerfas = self.aerfas.to(self.device)
        self.aerfa_ba = self.aerfa_ba.to(self.device)
        b,c,h,w = x.size()
        # 4 给每个样本随机生成时间步
        t = self.generate_t_steps(b).to(x.device)   
        index = t - 1
        # 5 计算每个样本添加t个时间布后的噪声图像
        Ɛ = torch.randn_like(x).to(x.device)
        sqrt_alpha_hat = torch.sqrt(self.aerfa_ba[index])[:, None, None, None]    # 第t个参数应该在index = t-1的位置拿取
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.aerfa_ba[index])[:, None, None, None]
        x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        # 6 unet预测噪声
        predicted_noise = self.unet(x, t)
        output = {
            "predict": predicted_noise,
            "label":Ɛ
        }
        return output
        
    def trainning(
            self,
            train_dataloader:DataLoader = None,
            test_dataloader:DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 1e-4,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            save_checkpoint_step:str = 10,
            save_model_dir:str = "models"
        ):
            ## 1 trainning log path 
            first_trainning = True
            check_point_path = save_model_dir  + "/checkpoint.pth"
            log_path = save_model_dir + "/train.log"

            ## 2 get net pretrain parameters if need 
            """
                If there is  training history record, load pretrain parameters
            """
            if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
                self.load_pretrained(save_model_dir)  
                first_trainning = False

            else:
                if not os.path.isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                with open(log_path, "w") as file:
                    pass

            ##  3 get optimizer
            if optimizer_name == "Adam":
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            elif optimizer_name == "AdamW":
                optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, 
                mode = "min", 
                factor = factor, 
                patience = patience
            )

            ## init trainng log
            if first_trainning:
                best_loss = float("inf")
                last_epoch = 0
            else:
                checkpoint = torch.load(check_point_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_loss = checkpoint["loss"]
                last_epoch = checkpoint["epoch"] + 1

            try:
                for epoch in range(last_epoch,total_epoch):
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = train_loss + test_loss
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": None,
                            "lr_scheduler": None
                        },
                        check_point_path
                    )

                    if epoch % save_checkpoint_step == 0:
                        os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                        torch.save(
                            {
                                "epoch": epoch,
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict()
                            },
                            save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                        )
                    if is_best:
                        self.save_pretrained(save_model_dir)

            # interrupt trianning
            except KeyboardInterrupt:
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        check_point_path
                    )

    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f}, use_memory: {:.1f}G".format(
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())

            str = "Test Epoch: {:d}, total_loss: {:.4f}".format(
                epoch,
                total_loss.avg, 
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg

    def compute_loss(self, input):
        output = {}
        mse_loss = nn.MSELoss()

        output["total_loss"] = mse_loss(input["predict"], input["label"])
        return output

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")    
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    unet = Unet(in_c=3,out_c=3,time_dim=32).to(device)
    x = torch.randn(batch, 3, 720, 960).to(device)
    t = torch.rand(batch).to(device)
    output_t = unet(x,t)
    print(output_t.shape)

