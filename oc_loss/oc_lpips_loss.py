import torch.nn as nn
import torch.nn.functional as F
import lpips

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

class OCLPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net_lpips = lpips.LPIPS(net="vgg").to(device)
        self.net_lpips.requires_grad_(False)
    
    def forward(self, x, y):
        # [-1, 1] -> [0, 1]
        x = OC(x * 0.5 + 0.5)
        y = OC(y * 0.5 + 0.5)
        loss = self.net_lpips(x, y)
        return loss
