import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn.utils import spectral_norm
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dino_gan.dinov3_vit_attention import SelfAttentionBlock

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

cur_path = "dino_gan"
class DINOv3ViT(nn.Module):
    def __init__(self, dinov3_vit_size='dinov3_vits16', blocks_to_take=[2, 5, 8]):
        super().__init__()
        dinov3_vit_weights = {
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
        }
        assert dinov3_vit_size in dinov3_vit_weights.keys(), f'`dinov3_vit_size` must be in {dinov3_vit_weights.keys()}'
        self.blocks_to_take = blocks_to_take
        self.model = torch.hub.load(
            f'{cur_path}/facebookresearch_dinov3_main', dinov3_vit_size, 
            source='local', weights=f"{cur_path}/{dinov3_vit_weights[dinov3_vit_size]}")  
        self.model.eval()
        self.model.requires_grad = False

        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )
    
    def _get_intermediate_layers(
        self,
        x: torch.Tensor,
        reshape: bool = False,
        norm: bool = True
    ) :
        outputs = []
        x, (H, W) = self.model.prepare_tokens_with_masks(x)
        for i, blk in enumerate(self.model.blocks):
            if self.model.rope_embed is not None:
                rope_sincos = self.model.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in self.blocks_to_take:
                outputs.append(x)
            if i == self.blocks_to_take[-1]: break 

        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        outputs = [out[:, self.model.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.model.patch_size, w // self.model.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        return outputs
    
    def forward(self, x):
        x = x * 0.5 + 0.5   # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        feats = self._get_intermediate_layers(x)
        return feats

def TokenBlurPool(channels, tokens_hw, filt_size=5, stride=1):
    return nn.Sequential(
        Rearrange('b (h w) c -> b c h w', h=tokens_hw, w=tokens_hw),
        BlurPool(channels=channels, filt_size=filt_size, stride=stride),
        Rearrange('b c h w -> b (h w) c'),
    )

def DownsampleLinear(dim, out_dim, token_hw):
    return nn.Sequential(
        Rearrange('b (h w) c -> b h w c', h=token_hw, w=token_hw),
        Rearrange('b (h s1) (w s2) c -> b (h w) (c s1 s2)', s1=2, s2=2),
        spectral_norm(nn.Linear(dim * 4, out_dim, 1))
    )

class MultiLevelViTDiscHead(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, num_level=3, resolution=1024):
        super().__init__()
        assert resolution % patch_size == 0, "Image resoultion must be divided by 16"
        self.num_level = num_level
        tokens_hw = resolution // 16
        dim_list = [embed_dim, embed_dim // 2, embed_dim // 4, embed_dim // 8] if resolution == 1024 else [embed_dim, embed_dim // 2, embed_dim // 4]

        self.multi_decoders = nn.ModuleList()
        for _ in range(num_level):  
            decoder = []     
            cur_tokens_hw = tokens_hw
            for i in range(len(dim_list) - 1):
                decoder.append(DownsampleLinear(dim_list[i], dim_list[i + 1], cur_tokens_hw))
                decoder.append(nn.LeakyReLU(0.2, inplace=True))
                decoder.append(TokenBlurPool(channels=dim_list[i + 1], tokens_hw=cur_tokens_hw // 2))
                if i != len(dim_list) - 2:
                    decoder.append(SelfAttentionBlock(dim=dim_list[i + 1], num_heads=dim_list[i + 1] // 64))
                cur_tokens_hw = cur_tokens_hw // 2
            decoder.append(spectral_norm(nn.Linear(dim_list[-1], 1)))
            self.multi_decoders.append(nn.Sequential(*decoder))

    def forward(self, x):
        logits = [self.multi_decoders[i](x[i]) for i in range(self.num_level)]
        return logits

class MultiLevelBCELoss(torch.nn.Module):
    def __init__(self, alpha=0.8, initial_weights=None):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        if initial_weights is not None:
            self.level_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True))
        else:
            self.level_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.6], dtype=torch.float32, requires_grad=True))
            
    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            for_real = True
        if for_real:
            target = self.alpha * torch.tensor(1.)
        else:
            target = torch.tensor(0.)
        
        loss_list = []
        for each in input:
            target_ = target.expand_as(each).to(each.device)
            loss_each = self.lossfn(each, target_)
            
            if len(loss_each.size()) > 1:
                loss_each = loss_each.mean(dim=tuple(range(1, len(loss_each.size()))))
            
            loss_list.append(loss_each.unsqueeze(1))  
        
        loss_per_level = torch.cat(loss_list, dim=1)  # [batch_size, num_levels]
        weights = torch.softmax(self.level_weights.unsqueeze(0), dim=-1)
        weighted_loss = loss_per_level * weights  
        total_loss = weighted_loss.sum(dim=1) 
        
        return total_loss

class Dinov3ViTDiscriminator(nn.Module):
    def __init__(self, resolution, dinov3_vit_size="base", diffaug=True, blocks_to_take=[2, 5, 8]):
        super().__init__() 
        self.dino = DINOv3ViT(dinov3_vit_size, blocks_to_take=blocks_to_take) 
        self.dino.requires_grad_(False) 
        embed_dim = self.dino.embed_dim
        patch_size = self.dino.patch_size
        self.decoder = MultiLevelViTDiscHead(
            embed_dim=embed_dim, patch_size=patch_size, num_level=len(blocks_to_take), resolution=resolution)
        self.decoder.requires_grad_(True)
        self.lossfn = MultiLevelBCELoss(0.8)
        self.lossfn.requires_grad_(True)
        if diffaug:
            self.policy = 'color,translation,cutout'

    def forward(self, x, for_real=True, for_G=False):
        x = DiffAugment(x, self.policy)
        feats = self.dino(x)
        logits = self.decoder(feats)
        loss = self.lossfn(logits, for_real=for_real, for_G=for_G)
        return loss



# # 使用示例
# if __name__ == "__main__":
    # 假设输入是3个层级的特征，每个尺寸为 [B, 768, 64, 64]
    # batch_size = 4
    # dummy_features = [
    #     torch.randn(batch_size, 768, 64, 64),
    #     torch.randn(batch_size, 768, 64, 64),
    #     torch.randn(batch_size, 768, 64, 64)
    # ]
    
    # # 初始化判别器
    # from dino_loss.vit import SelfAttentionBlock
    # # discriminator = MultiLevelVit()
    # attn = SelfAttentionBlock(
    #             dim=384,
    #             num_heads=6,
    #             ffn_ratio=4,
    #             qkv_bias=True,
    #             proj_bias=True,
    #             ffn_bias=True,
    #             drop_path=0.0,
    #             init_values=None,
    #             mask_k_bias=False,
    #             device="cuda",
    #         )
    # # # 前向传播
    # # output = discriminator(dummy_features)
    # # print(f"输出尺寸: {output.shape}")  # 应该是 [4, 1, 64, 64]
    
    # # 计算参数数量
    # total_params = sum(p.numel() for p in attn.parameters())
    # print(f"总参数数量: {total_params}")




# x = [
#     torch.randn(1, 192, 256, 256).to("cuda"),
#     torch.randn(1, 384, 128, 128).to("cuda"),
#     torch.randn(1, 768, 64, 64).to("cuda"),
#     torch.randn(1, 1536, 32, 32).to("cuda"),

# ]
# m = MultiLevelViTDiscHead().to("cuda")

# print(        f"Total vae_encoder training parameters: {sum([p.numel() for p in m.parameters() if p.requires_grad]) / 1000000} M"
# )
# print(m)
# z = m(x)
# for i in z:
#     print(i.shape)




# x = [
#     torch.randn(1, 1024, 768).to("cuda"),
#     torch.randn(1, 1024, 768).to("cuda"),
#     torch.randn(1, 1024, 768).to("cuda"),
# ]
# m = MultiLevelViTDiscHead(resolution=512).to("cuda")
# print(        f"Total vae_encoder training parameters: {sum([p.numel() for p in m.parameters() if p.requires_grad]) / 1000000} M"
# )
# print(m)
# z = m(x)
# for i in z:
#     print(i.shape)

# x = torch.randn(1, 4096, 768).to("cuda:0")
# d = DownsampleLinear(768, 384, 64).to("cuda:0")
# z = d(x)
# print(z.shape)
