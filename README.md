<div align="center">
<h2>OMGSR: You Only Need One Mid-timestep Guidance for Real-World Image Super-Resolution</h2>
<a href='https://arxiv.org/pdf/2508.08227'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

Zhiqiang Wu<sup>1,2*</sup> |
Zhaomang Sun<sup>2</sup> | 
Tong Zhou<sup>2</sup> | 
Bingtao Fu<sup>2</sup> | 
Ji Cong<sup>2</sup> |
Yitong Dong<sup>2</sup> |
\
Huaqi Zhang<sup>2</sup> |
Xuan Tang<sup>1</sup> |
Mingsong Chen<sup>1</sup> |
Xian Wei<sup>1&dagger;</sup> 

<sup>1</sup>Software Engineering Institute, East China Normal University | 
<sup>2</sup>vivo Mobile Communication Co. Ltd, Hangzhou, China |
<sup>*</sup>Work done during internship at vivo | 
<sup>&dagger;</sup>Corresponding author
</div>

## :boom: News
- **2025.9.15**: :boom: **Release the most powerful version (turbo version): `DINOv3-Convnext DISTS loss` and `DINOv3-Vit Discriminator`. Both support 1k-resolution GAN training. Please switch to the `turbo` branch.**
- **2025.8.25**: :hugs: We are replacing the VGG-based loss with the DINOv3-based loss. We are testing for excellent results.
- **2025.8.21**: :hugs: The pre-computed scripts of optimal mid-timestep are released.
- **2025.8.19**: :hugs: Integrate the DINOv3 GAN loss (support 512, 1k, or higher resolution for training).
- **2025.8.16**: :hugs: The training code is released.
- **2025.8.15**: The inference code and weights are released.
- **2025.8.12**: The arXiv paper is released.
- **2025.8.6**: This repo is released.

## :raised_hand_with_fingers_splayed: Note
- OMGSR-F-1024 can also be applied to face restoration, demonstrating excellent real skin texture (due to the training of GAN). Under 1k-resolution training, training only a few thousand steps (dual h20s with 4 gradient accumulation) yields good results.
- OMGSR-S-512 requires `~21G/24G` VRAM with `train_batch_size=1` and `gradient_checkpointing=True/False`.
- OMGSR-F-512 requires `~42G` VRAM with `train_batch_size=1` and `gradient_checkpointing=True`.
- OMGSR-F-1024 requires `~79G` VRAM with `train_batch_size=1` and `gradient_checkpointing=True`.
- If OMGSR is helpful to you, you could :star: this repo.

## :runner: TODO
- [x] ~~Release the inference code~~
- [x] ~~Release the weight~~
- [x] ~~Release the training code~~
- [x] ~~Integrate the DINOv3 GAN loss~~
- [ ] Develop OMGSR-Q (Qwen-Image) .....


## :fire: Training and inference pipelines

![teaser_img](assets/arch.png)

## :wrench: Environment

```
# git clone this repository
git clone https://github.com/wuer5/OMGSR.git
cd OMGSR
# create an environment
conda create -n OMGSR python=3.10
conda activate OMGSR
pip install --upgrade pip
pip install -r requirements.txt
```

## :rocket: Quick Inference





<h3>1. Download the DINOv3 weights </h3>

**The latest weight will release in these days!**

<h3>2. Download the LoRA weights </h3>

**The latest weight will release in these days!**

<h3>3. Start to inference</h3>

For OMGSR-S-512:
```bash
bash infer_omgsr_s.sh
```
For OMGSR-F-1024:
```bash
bash infer_omgsr_f.sh
```

## :hugs: Training 

<h3>1. Pre-compute your optimal mid-timestep</h3>
Note: Unlike the calculations in the paper, we revise the calculation formula. You can execute the command to obtain the relatively optimal mid-timestep.

For OMGSR-S-512:
```
python mid_timestep/mid_timestep_sd.py \
    --sd_path stabilityai/sd-turbo \
    --dataset_txt_or_dir_paths xxx xxx \
    --resolution 512
```
For OMGSR-F-512:
```
python mid_timestep/mid_timestep_flux.py \
    --flux_path black-forest-labs/FLUX.1-dev \
    --dataset_txt_or_dir_paths xxx xxx \
    --resolution 512
```
For OMGSR-F-1024:
```
python mid_timestep/mid_timestep_flux.py \
    --flux_path black-forest-labs/FLUX.1-dev \
    --dataset_txt_or_dir_paths xxx xxx \
    --resolution 1024
```

<h3>2. Prepare your training datasets</h3>

You should download the training datasets ```LSDIR``` and ```FFHQ``` (first 10k images) followed by our paper settings or your custom datasets.

You need to edit ```dataset_txt_or_dir_paths``` in the ```configs/xxx.yml``` like:

```
dataset_txt_or_dir_paths: [path1, path2, ...]
```
Note that ```path1, path2, ...``` can be the ```.txt``` path  (containing the paths of training images)  or the ```folder``` path (containing the training images). The type of images can be ```png, jpg, jpeg```.

<h3>3. Start to train</h3>

Please edit ```configs/xxx.yml``` for your own configuration.
Note: Do not forget to edit ```mid_timestep: [your optimal mid-timestep]``` in ```configs/xxx.yml```. Also, if you train OMGSR in ```[your optimal mid-timestep]```, do not forget to add ```--mid_timestep [your optimal mid-timestep]``` in ```infer_omgsr_x.sh``` during inference.

Start to train OMGSR-S at 512-resolution:
```
bash train_omgsr_s_512.sh
```

Start to train OMGSR-F at 512-resolution:
```
bash train_omgsr_f_512.sh
```

Start to train OMGSR-F at 1k-resolution:
```
bash train_omgsr_f_1024.sh
```


## :book: Citation

If OMGSR is helpful to you, you can cite this paper.
```bibtex
@misc{wu2025omgsrneedmidtimestepguidance,
      title={OMGSR: You Only Need One Mid-timestep Guidance for Real-World Image Super-Resolution}, 
      author={Zhiqiang Wu and Zhaomang Sun and Tong Zhou and Bingtao Fu and Ji Cong and Yitong Dong and Huaqi Zhang and Xuan Tang and Mingsong Chen and Xian Wei},
      year={2025},
      eprint={2508.08227},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.08227}, 
}
```
## :thumbsup: Acknowledgement
The `dino_gan` folder refer to <a href="https://github.com/nupurkmr9/vision-aided-gan">Vision-aided GAN</a> and <a href="https://github.com/facebookresearch/dinov3">DINOv3</a>. Thanks for these awesome works.

## :email: Contact

If you have any questions, please contact 51265902095@stu.ecnu.edu.cn.

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wuer5/OMGSR)
