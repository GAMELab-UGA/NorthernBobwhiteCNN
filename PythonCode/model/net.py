"""Defines the neural network, losss function and metrics"""

from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score
import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from local_context_norm import LocalContextNorm


class Conv(nn.Module):
    """(pointwise conv => depthwise conv => GroupNorm => GELU)"""
    
    def __init__(self, in_channels, out_channels, height):
        super().__init__()

        self.freq_bias = nn.Parameter(torch.zeros(height).view(1,1,-1,1))
        self.conv1 = nn.Conv2d(in_channels+1, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False, padding_mode='replicate')
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

        nn.init.orthogonal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        x = torch.cat((x, self.freq_bias.tile(x.shape[0],1,1,x.shape[-1])), 1)
        return self.act(self.norm(self.conv2(self.conv1(x))))


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, height):
        super().__init__()

        self.layer = Conv(in_channels, out_channels, height)

    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, height):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            Conv(in_channels, out_channels, height)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, height):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = Conv(in_channels, out_channels, height)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='replicate')
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, -2)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, height=128):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.height = height

        self.stem = Stem(n_channels, 32, height)

        self.down1 = Down(32, 32, height//2)
        self.down2 = Down(32, 32, height//4)
        self.down3 = Down(32, 32, height//8)
        self.down4 = Down(32, 32, height//16)

        self.up1 = Up(64, 32, height//8)
        self.up2 = Up(64, 32, height//4)
        self.up3 = Up(64, 32, height//2)
        self.up4 = Up(64 + n_channels, 32, height)

        self.head = Head(32, n_classes)

    def forward(self, input):
        x1 = self.stem(input)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x1 = torch.cat([input, x1], dim=1) # merge input before final decoder layer
        x = self.up4(x, x1)

        logits = self.head(x)
        return logits


class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()

        self.params = params

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=params.sample_rate,
            n_fft=params.window_size,
            win_length=params.window_size,
            hop_length=params.hop_length,
            f_min=params.fmin,
            f_max=params.fmax,
            pad=0,
            n_mels=params.n_mels,
            power=params.power,
            normalized=False,
            norm='slaney',
        )
        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=params.top_db)

        self.norm = LocalContextNorm(1, channels_per_group=1, window_size=(41,41), eps=1e-5)

        self.unet = UNet(n_channels=1, n_classes=1, height=params.n_mels)


    def forward(self, batch):
        x = batch["input"].float() # (bs, time)

        with autocast(enabled=False):
            mel_power_spec = self.mel_spec(x).unsqueeze(1) # (bs, 1, mel, time)
            mel_spec = self.amplitude_to_db(mel_power_spec) # amplitude to dB
            mel_spec = self.norm(mel_spec) # normalisation
            mel_spec = torch.nan_to_num(mel_spec)

        output_dict = {}
        output_dict['mel_spec'] = mel_spec

        # pad so temporal dim is divisible by 16
        time = mel_spec.shape[-1]
        extra = time % 16
        if extra > 0:
            pad = (16 - extra)
            mel_spec = F.pad(mel_spec, (0,pad,0,0), "replicate") # pad right

        # U-Net
        logits = self.unet(mel_spec) # (bs, 1, mel, time)

        # remove any padding from output
        logits = logits[:,:,:,:time]

        output_dict['logits'] = logits
        return output_dict


def precision(y_true, y_scores):
    return precision_score(y_true, y_scores)


def recall(y_true, y_scores):
    return recall_score(y_true, y_scores)


def average_precision(y_true, y_scores):
    return average_precision_score(y_true, y_scores)


def fbeta(y_true, y_scores, beta=1.0):
    return fbeta_score(y_true, y_scores, beta=beta)


def classification_loss(input, target, weight=None, beta=0.95, label_smoothing=0.01):
    prob = input.sigmoid().detach()
    soft_target = beta * target + (1-beta) * prob
    soft_target = soft_target.clamp(min=label_smoothing, max=1-label_smoothing)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input, soft_target, weight=weight)
    return loss


def dice_loss(input, target, smooth=1.):
    input = torch.sigmoid(input).view(-1)
    target = target.view(-1)
    
    intersection = (input * target).sum()
    dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
    return 1. - dice


def localization_loss(input, target, weight=None, beta=0.95, label_smoothing=0.001):
    dice = dice_loss(input, target)
    return dice


metrics = {
    'average_precision': average_precision,
    'precision': precision,
    'recall': recall,
    'fbeta': fbeta,
}
