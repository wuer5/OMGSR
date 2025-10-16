import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn.utils import spectral_norm
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(1.*(filt_size-1)/2), int(1.*(filt_size-1)/2), int(1.*(filt_size-1)/2)]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

# Copy from `https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py`
# convnext_sizes = {
#     "tiny": dict(
#         depths=[3, 3, 9, 3],
#         dims=[96, 192, 384, 768],
#     ),
#     "small": dict(
#         depths=[3, 3, 27, 3],
#         dims=[96, 192, 384, 768],
#     ),
#     "base": dict(
#         depths=[3, 3, 27, 3],
#         dims=[128, 256, 512, 1024],
#     ),
#     "large": dict(
#         depths=[3, 3, 27, 3],
#         dims=[192, 384, 768, 1536],
#     ),
# }
cur_path = "dinov3_gan"
class DINOv3ConvNeXt(torch.nn.Module):
    def __init__(self, dinov3_convnext_size):
        super().__init__()
        dinov3_convnext_weights = {
            'tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
            'small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
            'base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
            'large':'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
        }
        assert dinov3_convnext_size in dinov3_convnext_weights.keys(), f'`dinov3_convnext_size` must be in {dinov3_convnext_weights.keys()}'
        self.model = torch.hub.load(
            repo_or_dir=f'{cur_path}/facebookresearch_dinov3_main', 
            model=f'dinov3_convnext_{dinov3_convnext_size}', 
            source='local',
            weights=f"{cur_path}/dinov3_weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth")  
        
        self.model.requires_grad_(False)
        self.model.eval()
        
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )
        self.chns = self.model.embed_dims
    
    def _get_intermediate_layers(self, x, nums_to_take=3):
        output = []
        for i in range(nums_to_take):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            output.append(x)    # B x C x H x W
        return output

    def forward(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean) / self.std
        # we just use the first three layers
        feats = self._get_intermediate_layers(x)
        return feats

class MultiLevelConvNeXtDiscHead(nn.Module):
    def __init__(self, channels=[192, 384, 768], resolution=512):
        super().__init__()
        self.level = len(channels)
        self.decoders = nn.ModuleList([
            self._create_decoder(ch, *config) 
            for ch, config in zip(channels, self._get_configs(resolution))
        ])
    
    def _get_configs(self, resolution):
        if resolution == 1024:
            return [
                ([0.5, 1.0, 1.0, 1.0], [1, 2, 2, 2]),  
                ([0.5, 0.5, 1.0, 1.0], [1, 1, 2, 2]),  
                ([0.5, 0.5, 0.5, 1.0], [1, 1, 1, 2]) 
            ]  
        else:
            return [
                ([0.5, 1.0, 1.0], [2, 2, 2]),  
                ([0.5, 0.5, 1.0], [1, 2, 2]),  
                ([0.5, 0.5, 0.5], [1, 1, 2])   
            ]
        
    def _create_decoder(self, base_ch, ch_ratios, strides):
        layers = []
        cur_ch = base_ch
        
        for ratio, stride in zip(ch_ratios, strides):
            layers.extend([
                BlurPool(cur_ch, pad_type='zero'),
                spectral_norm(nn.Conv2d(cur_ch, int(cur_ch * ratio), 3, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            cur_ch = int(cur_ch * ratio)
        
        layers.extend([
            BlurPool(cur_ch, pad_type='zero'),
            spectral_norm(nn.Conv2d(cur_ch, 1, 1))
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return [dec(feat).squeeze(1) for dec, feat in zip(self.decoders, x)]

class MultiLevelBCELoss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            for_real = True
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        loss = 0
        for _, each in enumerate(input):
            target_ = target.expand_as(each).to(each.device)
            loss_ = self.lossfn(each, target_)
            if len(loss_.size()) > 2:
                loss_ = loss_.mean([1, 2]).reshape(-1, 1)
            loss += loss_
        return loss

class Dinov3ConvNeXtDiscriminator(nn.Module):
    def __init__(self, dinov3_convnext_size, resolution, diffaug=True):
        super().__init__() 
        self.dinov3_convnext = DINOv3ConvNeXt(dinov3_convnext_size=dinov3_convnext_size)
        # we just use the first three layers
        self.decoders = MultiLevelConvNeXtDiscHead(self.dinov3_convnext.chns[:3], resolution)
        self.decoders.requires_grad_(True)
        self.decoders.train()
        self.lossfn = MultiLevelBCELoss(0.8)
        if diffaug:
            self.policy = 'color,translation,cutout'

    def forward(self, x, for_real=True, for_G=False):
        x = DiffAugment(x, self.policy)
        feats = self.dinov3_convnext(x)
        logits = self.decoders(feats)
        loss = self.lossfn(logits, for_real=for_real, for_G=for_G)
        return loss.mean()
    
