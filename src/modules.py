import torch
from torch import nn
from einops import rearrange 
import torch
import torch
import torch.nn as nn
import numpy as np

from src.utils import *

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, (3, 3), (1, 1), 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_sample(x)

class TimeEmbed(nn.Module):
    def __init__(self, time_dim, out_dim):
        super(TimeEmbed, self).__init__()
        self.time_embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim,out_dim)
        )
    def forward(self,x):
        x = self.time_embed_layer(x)
        return rearrange(x,"b c -> b c 1 1")
    
class LabelEmbed(nn.Module):
    def __init__(self, label_dim, out_dim):
        super(LabelEmbed, self).__init__()
        self.label_embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(label_dim, out_dim)
        )
    def forward(self,x):
        x = self.label_embed_layer(x)
        return rearrange(x,"b c -> b c 1 1")

class SelfAttation3D(nn.Module):
    def __init__(self, channels, h,w):
        super(SelfAttation3D, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.h * self.w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.h, self.w)

class Unet(nn.Module):
    def __init__(self, width,  height,  in_c = 3, out_c = 3, time_dim=32):
        super(Unet, self).__init__()
        self.h = height
        self.w = width
        self.time_dim = time_dim
        self.conv1 = DoubleConv(in_c, 64)
        self.att1 = SelfAttation3D(64,self.h,self.w)
        self.e_emd_1 = TimeEmbed(time_dim,64)
        self.pool1 = DownSample(64, 64)  # 1/2
        self.conv2 = DoubleConv(64, 128)
        self.att2 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.e_emd_2 = TimeEmbed(time_dim,128)
        self.pool2 = DownSample(128, 128)  
        self.conv3 = DoubleConv(128, 256)
        self.att3 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.e_emd_3 = TimeEmbed(time_dim,256)
        self.pool3 = DownSample(256, 256)  
        self.conv4 = DoubleConv(256, 512)
        self.att4 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.e_emd_4 = TimeEmbed(time_dim,512)
        self.pool4 = DownSample(512, 512)  

        self.conv5 = DoubleConv(512, 1024)

        self.up6 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))  
        self.conv6 = DoubleConv(1024, 512)
        self.att6 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.d_emd_1 = TimeEmbed(time_dim,512)

        self.up7 = torch.nn.ConvTranspose2d(512, 256, (2, 2), (2, 2))
        self.conv7 = DoubleConv(512, 256)
        self.att7 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.d_emd_2 = TimeEmbed(time_dim,256)

        self.up8 = torch.nn.ConvTranspose2d(256, 128, (2, 2), (2, 2))
        self.conv8 = DoubleConv(256, 128)
        self.att8 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.d_emd_3 = TimeEmbed(time_dim,128)

        self.up9 = torch.nn.ConvTranspose2d(128, 64, (2, 2), (2, 2))
        self.conv9 = DoubleConv(128, 64)
        self.att9 = SelfAttation3D(64,(int)(self.h),(int)(self.w))
        self.d_emd_4 = TimeEmbed(time_dim,64)

        self.conv10 = torch.nn.Conv2d(64, out_c, kernel_size=(1, 1))  
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc        
 

    def forward(self, x,t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        c1 = self.conv1(x) + self.e_emd_1(t)
        c1 = self.att1(c1)
        p1 = self.pool1(c1)  
        c2 = self.conv2(p1) + self.e_emd_2(t)
        c2 = self.att2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2) + self.e_emd_3(t)
        c3 = self.att3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3) + self.e_emd_4(t)
        c4 = self.att4(c4)
        p4 = self.pool4(c4) 

        c5 = self.conv5(p4)

        up6 = self.up6(c5)  
        m6 = torch.cat([up6, c4], dim=1)
        c6 = self.conv6(m6) + self.d_emd_1(t)
        c6 = self.att6(c6)

        up7 = self.up7(c6) 
        m7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(m7) + self.d_emd_2(t)
        c7 = self.att7(c7)

        up8 = self.up8(c7)  
        m8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(m8) + self.d_emd_3(t)
        c8 = self.att8(c8)

        up9 = self.up9(c8)  
        m9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(m9) + self.d_emd_4(t)
        c9 = self.att9(c9)

        c10 = self.conv10(c9)
        return c10
    
class CUnet(nn.Module):
    def __init__(self, width,  height,  in_c = 3, out_c = 3, time_dim = 512, label_dim = 10):
        super(CUnet, self).__init__()
        self.h = height
        self.w = width
        self.time_dim = time_dim
        self.label_dim = label_dim
        
        self.conv1 = DoubleConv(in_c, 64)
        self.att1 = SelfAttation3D(64,self.h,self.w)
        self.e_emd_1 = TimeEmbed(self.time_dim,64)
        self.l_emd_1 = LabelEmbed(self.label_dim, 64)
        self.pool1 = DownSample(64, 64)  
        
        self.conv2 = DoubleConv(64, 128)
        self.att2 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.e_emd_2 = TimeEmbed(self.time_dim, 128)
        self.l_emd_2 = LabelEmbed(self.label_dim, 128)
        self.pool2 = DownSample(128, 128)  
        
        self.conv3 = DoubleConv(128, 256)
        self.att3 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.e_emd_3 = TimeEmbed(self.time_dim, 256)
        self.l_emd_3 = LabelEmbed(self.label_dim, 256)
        self.pool3 = DownSample(256, 256)  
        
        self.conv4 = DoubleConv(256, 512)
        self.att4 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.e_emd_4 = TimeEmbed(self.time_dim, 512)
        self.l_emd_4 = LabelEmbed(self.label_dim, 512)
        self.pool4 = DownSample(512, 512)  

        self.conv5 = DoubleConv(512, 1024)

        self.up6 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))  
        self.conv6 = DoubleConv(1024, 512)
        self.att6 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.d_emd_1 = TimeEmbed(self.time_dim,512)
        self.ld_emd_1 = LabelEmbed(self.label_dim, 512)

        self.up7 = torch.nn.ConvTranspose2d(512, 256, (2, 2), (2, 2))
        self.conv7 = DoubleConv(512, 256)
        self.att7 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.d_emd_2 = TimeEmbed(self.time_dim,256)
        self.ld_emd_2 = LabelEmbed(self.label_dim, 256)

        self.up8 = torch.nn.ConvTranspose2d(256, 128, (2, 2), (2, 2))
        self.conv8 = DoubleConv(256, 128)
        self.att8 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.d_emd_3 = TimeEmbed(self.time_dim,128)
        self.ld_emd_3 = LabelEmbed(self.label_dim, 128)

        self.up9 = torch.nn.ConvTranspose2d(128, 64, (2, 2), (2, 2))
        self.conv9 = DoubleConv(128, 64)
        self.att9 = SelfAttation3D(64,(int)(self.h),(int)(self.w))
        self.d_emd_4 = TimeEmbed(self.time_dim,64)
        self.ld_emd_4 = LabelEmbed(self.label_dim, 64)

        self.conv10 = torch.nn.Conv2d(64, out_c, kernel_size=(1, 1))  
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc        
 

    def forward(self, x, t, label):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        label = nn.functional.one_hot(label, num_classes=10).type(torch.float)
        # label = label.unsqueeze(-1).type(torch.float)
        # label = self.pos_encoding(label, self.label_dim)
        c1 = self.conv1(x) *self.l_emd_1(label)  + self.e_emd_1(t)
        c1 = self.att1(c1)
        p1 = self.pool1(c1)  
        c2 = self.conv2(p1) * self.l_emd_2(label) + self.e_emd_2(t)
        c2 = self.att2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2) * self.l_emd_3(label) + self.e_emd_3(t)
        c3 = self.att3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3) * self.l_emd_4(label) + self.e_emd_4(t)
        c4 = self.att4(c4)
        p4 = self.pool4(c4) 

        c5 = self.conv5(p4)

        up6 = self.up6(c5)  
        m6 = torch.cat([up6, c4], dim=1)
        c6 = self.conv6(m6) * self.ld_emd_1(label) + self.d_emd_1(t)
        c6 = self.att6(c6)

        up7 = self.up7(c6) 
        m7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(m7) * self.ld_emd_2(label) + self.d_emd_2(t) 
        c7 = self.att7(c7)

        up8 = self.up8(c7)  
        m8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(m8) * self.ld_emd_3(label) + self.d_emd_3(t) 
        c8 = self.att8(c8)

        up9 = self.up9(c8)  
        m9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(m9) * self.ld_emd_4(label) + self.d_emd_4(t)
        c9 = self.att9(c9)

        c10 = self.conv10(c9)
        return c10