import torch
import torch.nn as nn
import pyiqa
import torch.nn.functional as F

# Overlap-Chunked
def OC(x, patch_size=224):
    if x.size(-1) == 512 and patch_size == 518: return x
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

    def _rgb_to_grayscale(self, rgb_img):
        """Convert RGB image to grayscale using weighted average"""
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb_img.device).view(1, 3, 1, 1)
        return (rgb_img * rgb_weights).sum(dim=1, keepdim=True)

    def forward(self, x, eps=1e-6):
        if x.dim() == 4 and x.size(1) == 3:
            x = self._rgb_to_grayscale(x)
        x = self.sobel_conv(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + eps)
        return x

class OCEADISTSLoss(nn.Module):
    def __init__(self, device):
        super(OCEADISTSLoss, self).__init__()    

        # dists loss using pyiqa
        self.dists_loss = pyiqa.create_metric('dists', device=device, as_loss=True)
        # sobel
        self.sobel_operator = Sobel().to(device)
        self.sobel_operator.requires_grad_(False)

    # def forward(self, x, y):
    #     # [-1, 1] -> [0, 1]
    #     x = OC(x * 0.5 + 0.5)
    #     y = OC(y * 0.5 + 0.5)
    #     d_loss = self.dists_loss(x, y)
    #     edge_x = self.sobel_operator(x)
    #     edge_y = self.sobel_operator(y)
    #     e_loss = self.dists_loss(edge_x, edge_y)
    #     total_loss = d_loss + e_loss
    #     return total_loss
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
