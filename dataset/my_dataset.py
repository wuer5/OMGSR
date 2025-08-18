import sys
import glob
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.realesrgan import RealESRGAN_degradation

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_txt_or_dir_paths, resolution):
        super().__init__()
        self.resolution = resolution
        self.degradation = RealESRGAN_degradation(device='cpu', resolution=resolution)
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop(
                (resolution, resolution), 
                pad_if_needed=True,
                padding_mode='reflect'
            ),
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
        ])
        self.gt_list = []
        for p in dataset_txt_or_dir_paths:
            if os.path.isdir(p):
                self.gt_list.extend(glob.glob(f"{p}/*.png") + glob.glob(f"{p}/*.jpg") + glob.glob(f"{p}/*.jpeg"))
            elif os.path.splitext(p)[1] == ".txt":
                with open(p, 'r') as f:
                    self.gt_list.extend([line.strip() for line in f.readlines()])
            else:
                raise ValueError(f"Unsupported path type: {p}. Expected either a directory or a file named 'txt'")
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_path = self.gt_list[idx]
        gt_img = Image.open(gt_path).convert('RGB')
        if 'ffhq' in gt_path and self.resolution == 512:   
            gt_img = gt_img.resize((512, 512), Image.Resampling.LANCZOS)  
        gt_img = self.crop_preproc(gt_img)

        img_gt, img_lq = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
        img_gt, img_lq = img_gt.squeeze(0), img_lq.squeeze(0)

        # input images scaled to -1,1
        img_gt = F.normalize(img_gt, mean=[0.5], std=[0.5])
        # output images scaled to -1,1
        img_lq = F.normalize(img_lq, mean=[0.5], std=[0.5])

        return img_lq, img_gt
            
   