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

## :boom: HighLight
Unlike the paper, this repo has been further optimized by: 

- Replace *LPIPS Loss (natively support 224 resolution)* with the proposed ***DINOv3-ConNext DISTS Loss (natively support 1k or higher resolution)*** for structural perception.

- Develop ***DINOv3-ConNext Multi-level Discriminator Head (natively support 1k or higher resolution)*** for GAN training.


## :boom: News
- **2025.10.14**: :hugs: **The latest version is released.**
- **2025.8.16**: The training code is released.
- **2025.8.15**: The inference code and weights are released.
- **2025.8.12**: The arXiv paper is released.
- **2025.8.6**: This repo is released.


## :eyes: Visualization
**TODO**

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

<h3>1. Download the pre-trained models from huggingface</h3>

- Download <a href="https://huggingface.co/stabilityai/stable-diffusion-2-1">SD2.1-Base</a> for OMGSR-S-512.
- Download <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">FLUX.1-dev</a> for OMGSR-F-1024.
<h3>2. Download the OMGSR Lora adapters weights</h3>

**TODO**

<h3>3. Prepare your testing data</h3>

You should put the testing data (```.png```, ```.jpg```, ```.jpeg``` formats) to the folder ```tests```.

<h3>4. Start to inference</h3>

For OMGSR-S-512:
```bash
bash infer_omgsr_s.sh
```
For OMGSR-F-1024:
```bash
bash infer_omgsr_f.sh
```

## :hugs: Training 

<h3>1. Prepare your training datasets</h3>

You should download the training datasets ```LSDIR``` and ```FFHQ``` (first 10k images) followed by our paper settings or your custom datasets.

You need to edit ```dataset_txt_or_dir_paths``` in the ```configs/xxx.yml``` like:

```
dataset_txt_or_dir_paths: [path1, path2, ...]
```
Note that ```path1, path2, ...``` can be the ```.txt``` path  (containing the paths of training images)  or the ```folder``` path (containing the training images). The type of images can be ```png, jpg, jpeg```.


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

If OMGSR is helpful to you, you could cite this paper.
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
The `dinov3_gan` folder in this project is modified from <a href="https://github.com/nupurkmr9/vision-aided-gan">Vision-aided GAN</a> and <a href="https://github.com/facebookresearch/dinov3">DINOv3</a>. Thanks for these awesome work.

## :email: Contact

If you have any questions, please contact 51265902095@stu.ecnu.edu.cn.

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wuer5/OMGSR)