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
    
cur_path = "dino_gan"
class DINOv3Convnext(torch.nn.Module):
    def __init__(self, dino_convnext_size="large"):
        super().__init__()
        dino_convnext_weights = {
            'tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
            'small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
            'base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
            'large':'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
        }
        assert dino_convnext_size in dino_convnext_weights.keys(), f'`dino_convnext_size` must be in {dino_convnext_weights.keys()}'

        self.model = torch.hub.load(
            f'{cur_path}/facebookresearch_dinov3_main', f'dinov3_convnext_{dino_convnext_size}', source='local',
            weights=f"{cur_path}/dino_weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth")  
        self.model.eval()
        self.model.requires_grad = False
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )
        self.chns = self.model.embed_dims
        self.pools = nn.ModuleList([
            L2pooling(channels=self.chns[i], filter_size=5, stride=1) 
            for i in range(len(self.chns))
        ])

    def _get_intermediate_layers(self, x, n=1):
        output, total_block_len = [], len(self.model.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            x = self.pools[i](x)
            if i in blocks_to_take:
                output.append(x)    # B x C x H x W
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def __call__(self, x):
        x = x * 0.5 + 0.5
        h = (x - self.mean) / self.std
        feats = self._get_intermediate_layers(h, n=4)
        return [x] + feats

class DINOv3ConvnextDISTS(torch.nn.Module):
    def __init__(self, dino_convnext_size="large"):
        super().__init__()
        self.dino = DINOv3Convnext(dino_convnext_size)
        self.dino.requires_grad_(False)
        self.chns = [3] + self.dino.chns
        self.init_value = 1 / (2 * sum(self.chns))

    def forward(self, x, y):
        feats0 = self.dino(x)
        feats1 = self.dino(y)
        dist1 = dist2 = 0
        c1 = c2 = 1e-6

        for k in range(len(self.chns)):
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

        return score.squeeze(-1).squeeze(-1)
    




