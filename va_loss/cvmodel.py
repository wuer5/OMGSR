import importlib
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
import clip
import antialiased_cnns
import gdown
from torchvision import transforms
from va_loss.DiffAugment_pytorch import DiffAugment

cur_path = os.path.dirname(os.path.abspath(__file__))

class Vgg(nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model = antialiased_cnns.vgg16(pretrained=True, filter_size=4).features
        self.model.eval()
        self.model.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
        
    def forward(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
        out = self.model(x)

        if 'pool' in self.cv_type:
            out = self.pool(out)
            out = out.view(out.shape[0], -1)

        return out


class Swin(torch.nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model = timm.create_model('swin_tiny_patch4_window7_224')
        if not os.path.exists(os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth')):
            st = gdown.download("https://drive.google.com/u/0/uc?id=1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u&export=download", os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth'))

        st = torch.load(os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth'), map_location='cpu')
        new_st = {}
        for each in st['model'].keys():
            if 'encoder.' in each:
                newk = each.replace('encoder.', '')
                new_st[newk] = st['model'][each]
        self.model.load_state_dict(new_st, strict=False)

        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_custom(self, x, return_intermediate=False):
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)     
        x = self.model.norm(x)
        if return_intermediate:
            return x.transpose(1, 2)
        
        x = x.mean(dim=1)
        return x

    def __call__(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv' in self.cv_type:
            x = self.forward_custom(x, return_intermediate=True)
            x = x.reshape(-1, 768, 7, 7)
            return x
            
        return self.forward_custom(x)


class CLIP(torch.nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model, _ = clip.load("ViT-B/32", jit=False, device='cpu')
        self.model = self.model.visual
        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
 
    def forward_custom(self, x):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x1 = []
        feat_points = [0, 4, 8, len(self.model.transformer.resblocks)]
        for i in range(len(feat_points)-1):
            x = self.model.transformer.resblocks[feat_points[i]:feat_points[i+1]](x)
            x1.append(x.permute(1, 0, 2))

        x = self.model.ln_post(x1[-1][:, 0, :])
        if self.model.proj is not None:
            x = x @ self.model.proj
        x1[-1] = x
        return x1

    def __call__(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
            
        if 'conv_multi_level' in self.cv_type:
            x = self.forward_custom(x.type(self.model.conv1.weight.dtype))
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[2] = x[2].float()
        else:
            x = self.model(x.type(self.model.conv1.weight.dtype)).float()
            
        return x
    
    
class DINO(torch.nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = 224
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
       
    def __call__(self, x):
        if x.size(-1) == self.input_resolution:
            x = x * 0.5 + 0.5
        else:
            x = F.interpolate(x * 0.5 + 0.5, size=(self.input_resolution, self.input_resolution), mode='area')
        x = self.normalize(x)
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, self.nums_patchs, self.nums_patchs)
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, self.nums_patchs, self.nums_patchs)
            x[2] = x[2][:, 0, :]
        else:
            x = self.model(x)
        return x

# add DINOv2 for 518 resolution
class DINOv2(torch.nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')  # without cls token
        self.model = torch.hub.load(f'{cur_path}/facebookresearch_dinov2_main', 'dinov2_vitb14', source='local')  # without cls token
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = 518 
        self.nums_patchs = self.input_resolution // 14
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
       
    def __call__(self, x):
        if x.size(-1) == self.input_resolution:
            x = x * 0.5 + 0.5
        else:
            x = F.interpolate(x * 0.5 + 0.5, size=(self.input_resolution, self.input_resolution), mode='area')
        x = self.normalize(x)
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0].permute(0, 2, 1).reshape(-1, 768, self.nums_patchs, self.nums_patchs)   # 518 // 14 = 37
            x[1] = x[1].permute(0, 2, 1).reshape(-1, 768, self.nums_patchs, self.nums_patchs)
            # x[2] = x[2]
        else:
            x = self.model(x)
            
        return x

# add DINOv3 for 512, 1k or higher resolution
class DINOv3(torch.nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()

        self.cv_type = cv_type
        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')  # without cls token
        self.model = torch.hub.load(f'{cur_path}/facebookresearch_dinov3_main', 'dinov3_vitb16', source='local', 
                                    weights=f'{cur_path}/dino_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')  # without cls token
        self.model.eval()
        self.model.requires_grad = False
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
    def __call__(self, x):
        nums_patchs = x.size(-1) // 16
        # [-1, 1] -> [0, 1]
        x = x * 0.5 + 0.5
        x = self.normalize(x)
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0].permute(0, 2, 1).reshape(-1, 768, nums_patchs, nums_patchs)   
            x[1] = x[1].permute(0, 2, 1).reshape(-1, 768, nums_patchs, nums_patchs)
            # x[2] = x[2]
        else:
            x = self.model(x)
        return x

class CVBackbone(torch.nn.Module):
    def __init__(self, cv_type, output_type, diffaug=False, device='cpu'):
        super().__init__()
        cv_type = cv_type.split('+')
        output_type = output_type.split('+')
        self.class_name_dict = {
            'seg_ade': 'va_loss.swintaskspecific.Swin',
            'det_coco': 'va_loss.swintaskspecific.Swin',
            'clip': 'va_loss.cvmodel.CLIP',
            'dino': 'va_loss.cvmodel.DINO',
            'dinov2': 'va_loss.cvmodel.DINOv2',  # add here for DINOv2
            'dinov3': 'va_loss.cvmodel.DINOv3',  # add here for DINOv3
            'vgg': 'va_loss.cvmodel.Vgg',
            'swin': 'va_loss.cvmodel.Swin',
            'face_seg': 'va_loss.face_parsing.Parsing',
            'face_normals': 'va_loss.face_normals.Normals'
        }

        self.cv_type = cv_type
        self.policy = ''
        if diffaug:
            self.policy = 'color,translation,cutout'
            
        self.models = []
        for cv_type_, output_type_ in zip(cv_type, output_type):
            modellib = importlib.import_module('.'.join(self.class_name_dict[cv_type_].split('.')[:-1]))
            model = None
            target_model_name = self.class_name_dict[cv_type_].split('.')[-1]
            for name, cls in modellib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    model = cls
                    
            cv_type_ = cv_type_ + '_' + output_type_
            model = model(cv_type=cv_type_).requires_grad_(False).to(device)
            self.models.append(model)

    def __call__(self, images):
        image_features = []
        for i, each in enumerate(self.models):
            image_features.append(each(DiffAugment(images, policy=self.policy)))
        return image_features