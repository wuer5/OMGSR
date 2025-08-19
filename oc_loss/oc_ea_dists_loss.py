import os
from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

cur_path = os.path.dirname(os.path.abspath(__file__))

# Overlap-Chunked
def OC(x, patch_size=224):
    if len(x) == 3:
        x = x.unsqueeze(0)
    _, C, H, W = x.shape
    assert H == W
    if H % patch_size == 0:
        stride = patch_size
    else:
        N = int((H / patch_size)) + 1
        stride = patch_size - (N * patch_size - H) // (N - 1)
    patches = F.unfold(
        x, 
        kernel_size=patch_size, 
        stride=stride
    )  
    patches = patches.permute(0, 2, 1).reshape(
        -1, C, patch_size, patch_size
    )  # (num_patches, C, patch_size, patch_size)
    
    return patches

def clean_state_dict(state_dict):
    """
    Clean checkpoint by removing .module prefix from state dict if it exists from parallel training.

    Args:
        state_dict (dict): State dictionary from a model checkpoint.

    Returns:
        dict: Cleaned state dictionary.
    """
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict



class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
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

class DISTS(torch.nn.Module):
    r"""DISTS model.
    Args:
        pretrained_model_path (String): Pretrained model path.

    """

    def __init__(self, pretrained_model_path=None):
        """Refer to official code https://github.com/dingkeyan93/DISTS"""
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='IMAGENET1K_V1').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter(
            'alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.register_parameter(
            'beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        if pretrained_model_path is None:
            pretrained_model_path = f"{cur_path}/DISTS_weights-f5e65c96.pth"
        print(f'Loading pretrained model {self.__class__.__name__} from {pretrained_model_path}')
        state_dict = torch.load(
            pretrained_model_path, map_location=torch.device('cpu'), weights_only=False
        )
        state_dict = clean_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=False)

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y):
        r"""Compute IQA using DISTS model.

        Args:
            - x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            - y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DISTS model.

        """
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2)

        return score.squeeze(-1).squeeze(-1)



class Sobel(nn.Module):
    def __init__(self):
        super().__init__()

        self.sobel_conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        Gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        Gy = torch.tensor([[-1.,-2.,-1.],
                           [ 0., 0., 0.],
                           [ 1., 2., 1.]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.sobel_conv.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, x, eps=1e-6):
        if x.dim() == 4 and x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)
        x = self.sobel_conv(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + eps)
        return x

class OCEADISTSLoss(nn.Module):
    def __init__(self, device):
        super(OCEADISTSLoss, self).__init__()    

        # dists loss using pyiqa
        # self.dists_loss = pyiqa.create_metric('dists', device=device, as_loss=True)
        self.dists_loss = DISTS().to(device)
        # sobel
        self.sobel_operator = Sobel().to(device)
        self.sobel_operator.requires_grad_(False)

    def forward(self, x, y):
        # [-1, 1] -> [0, 1]
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        d_loss = self.dists_loss(OC(x), OC(y))
        edge_x = self.sobel_operator(x)
        edge_y = self.sobel_operator(y)
        e_loss = self.dists_loss(OC(edge_x), OC(edge_y))
        total_loss = d_loss + e_loss
        return total_loss
