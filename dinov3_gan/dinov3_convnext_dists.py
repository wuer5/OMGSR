import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class L2pooling(nn.Module):
    def __init__(self, channels, filter_size=5, stride=2):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            'filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, input):
        input = input**2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()

cur_path = "dinov3_gan"
class L2PooledDINOv3ConvNext(torch.nn.Module):
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
        self.l2pool_layers = nn.ModuleList([
            L2pooling(channels=ch, filter_size=5, stride=1) 
            for ch in self.chns[:3]
        ])

    def _get_l2pooled_intermediate_layers(self, x, nums_to_take=3):
        output = []
        for i in range(nums_to_take):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            x = self.l2pool_layers[i](x)  # Add L2pooling
            output.append(x)    # B x C x H x W
        return output

    def forward(self, x):
        x = x * 0.5 + 0.5
        h = (x - self.mean) / self.std
        # we just use the first three layers
        pooled_feats = self._get_l2pooled_intermediate_layers(h)
        return [x] + pooled_feats 

class DINOv3ConvNeXtDISTS(torch.nn.Module):
    def __init__(self, dinov3_convnext_size):
        super().__init__()
        self.l2pooled_dinov3_convnext = L2PooledDINOv3ConvNext(dinov3_convnext_size)        
        self.channels = [3] + self.l2pooled_dinov3_convnext.chns[:3]
        self.init_value = 1 / (2 * sum(self.channels))

    def forward(self, x, y):
        feats0 = self.l2pooled_dinov3_convnext(x)
        feats1 = self.l2pooled_dinov3_convnext(y)
        dist1 = dist2 = 0
        c1 = c2 = 1e-6

        for k in range(len(self.channels)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (self.init_value * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (self.init_value * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2)

        return score.mean()
