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

- Replace ~~*LPIPS Loss (natively support 224 resolution)*~~ with the proposed ***DINOv3-ConNext DISTS Loss (natively support 1k or higher resolution)*** for structural perception.

- Develop ***DINOv3-ConNext Multi-level Discriminator Head (natively support 1k or higher resolution)*** for GAN training.


## :boom: News
- **2025.10.14**: :hugs: **The latest version is released.**
- **2025.8.16**: The training code is released.
- **2025.8.15**: The inference code and weights are released.
- **2025.8.12**: The arXiv paper is released.
- **2025.8.6**: This repo is released.


## :eyes: Visualization
These demos are created by OMGSR-F-1024, which has demonstrated excellent representation, especially in terms of details.
**Click** the images for comparative visualization.

[<img src="assets/1.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDQ0)
[<img src="assets/2.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDQz)
[<img src="assets/3.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDQ4)
[<img src="assets/4.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDQ5)
[<img src="assets/5.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDUw)
<!-- https://imgsli.com/NDIyMDQ0
https://imgsli.com/NDIyMDQz
https://imgsli.com/NDIyMDQ4
https://imgsli.com/NDIyMDQ5
https://imgsli.com/NDIyMDUw -->

[<img src="assets/6.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDUy)
[<img src="assets/7.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDUz)
[<img src="assets/8.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDUx)
[<img src="assets/9.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDU0)
[<img src="assets/10.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDU1)
<!-- https://imgsli.com/NDIyMDUy
https://imgsli.com/NDIyMDUz
https://imgsli.com/NDIyMDUx
https://imgsli.com/NDIyMDU0
https://imgsli.com/NDIyMDU1 -->

[<img src="assets/11.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDU3)
[<img src="assets/12.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDU4)
[<img src="assets/13.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDU5)
[<img src="assets/14.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDYw)
[<img src="assets/13.png" target="_blank" height="160px"/>](https://imgsli.com/NDIyMDYx)
<!-- https://imgsli.com/NDIyMDU3
https://imgsli.com/NDIyMDU4
https://imgsli.com/NDIyMDU5
https://imgsli.com/NDIyMDYw
https://imgsli.com/NDIyMDYx -->

## Averge Optimal Mid-timestep via Signal-to-Noise Ratio (SNR)
#### 1. Pre-trained Noisy Latent Representation
$$
\text{DDPM}: \mathbf{z}_t
= \sqrt{\bar{\alpha}_t} \mathbf{z}_H + \sqrt{1-\bar{\alpha}_t} \epsilon.
\quad
\text{FM}: \mathbf{z}_t
= (1 - \sigma_t) \mathbf{z}_H + \sigma_t \epsilon.
$$
#### 2. SNR of Pre-trained Noisy Latent Representation
$$   
\text{DDPM}: \texttt{SNR}(\mathbf{z}_t)=\frac{\bar{\alpha}_t  \cdot \mathbb{E}[\mathbf{z}_{H}^2]}{(1 - \bar{\alpha}_t)  \cdot\mathbb{E}[\epsilon^2]}=\frac{\bar{\alpha}_t \cdot \mathbb{E}[\mathbf{z}_H^2]}{1 - \bar{\alpha}_t}.
\quad
\text{FM}: \texttt{SNR}(\mathbf{z}_t)=\frac{(1 - \sigma_t)^2  \cdot \mathbb{E}[\mathbf{z}_{H}^2]}{\sigma_t^2 \cdot \mathbb{E}[\epsilon^2]}=\frac{(1 - \sigma_t)^2 \cdot \mathbb{E}[\mathbf{z}_H^2]}{\sigma_t^2}.
$$
#### 3. SNR of Low-Quality (LQ) Image Latent Representation
$$
\texttt{SNR}(\mathbf{z}_L) = \frac{\mathbb{E}[\mathbf{z}_H^2]}{\mathbb{E}[(\mathbf{z}_L - \mathbf{z}_H)^2]}
$$

#### 4. Compute Averge Optimal Mid-timestep

$$ t^\ast = \arg \min_t \frac{1}{N}\sum_{i=1}^N \left|\text{SNR}(\mathbf{z}_t^{(i)}) - \text{SNR}(\mathbf{z}_L^{(i)})\right|, \quad \text{Dataset:} \\{(\mathbf{z}_L^{(i)}, \mathbf{z}_H^{(i)})\\}_N$$


#### 5. Mid-timestep Script
You can run the script:

```
# OMGSR-S-512
python mid_timestep/mid_timestep_sd.py --dataset_txt_or_dir_paths /path1/to/images /path2/to/images
```
```
# OMGSR-F-1024
python mid_timestep/mid_timestep_flux.py --dataset_txt_or_dir_paths /path1/to/images /path2/to/images
```
- In this repo, we using mid-timestep `273` for `OMGSR-S-512` and `244` for `OMGSR-F-1024`.
- In fact, a mid-timestep around the recommended value is also ok and does not need to be very accurate. 
- Note that the mid-timesteps during training and inference should be consistent
- The mid-timestep is actually related to degraded configuration in a dataset.


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

### 1. Download the pre-trained models from huggingface

- Download <a href="https://huggingface.co/stabilityai/stable-diffusion-2-1">SD2.1-base</a> for OMGSR-S-512.
- Download <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">FLUX.1-dev</a> for OMGSR-F-1024.

### 2. Download the OMGSR Lora adapter weights

- Download the <a href="#">OMGSR-S-512 Lora Adapter Weight</a> (rename it as `omgsr-s-512-adapter`) to the folder `adapters` (please make the folder). *[TODO]*

- Download the <a href="https://drive.google.com/drive/folders/11pPiyQ7YUpDmc5uyZ0x7uBB07bGBAbov?usp=sharing">OMGSR-F-1024 Lora Adapter Weight</a> (rename it as `omgsr-f-1024-adapter`) to the folder `adapters` (please make the folder).


### 3. Prepare your testing data

You should put the testing data (```.png```, ```.jpg```, ```.jpeg``` formats) to the folder ```tests```.

### 4. Start to inference

For OMGSR-S-512:
```bash
bash infer_omgsr_s.sh
```
For OMGSR-F-1024:
```bash
bash infer_omgsr_f.sh
```

## :hugs: Training 

### 1. Prepare your training datasets

You should download the training datasets ```LSDIR``` and ```FFHQ``` (first 10k images) followed by our paper settings or your custom datasets.

You need to edit ```dataset_txt_or_dir_paths``` in the ```configs/xxx.yml``` like:

```
dataset_txt_or_dir_paths: [path1, path2, ...]
```
Note that ```path1, path2, ...``` can be the ```.txt``` path  (containing the paths of training images)  or the ```folder``` path (containing the training images). The type of images can be ```png, jpg, jpeg```.

### 2. Download the DINOv3-ConvNext

You can download the <a href="https://drive.google.com/file/d/1-kSZ2BfBJfO4DvEftju__XGT6Rsj596m/view?usp=sharing">DINOv3-ConvNext-Large</a> to the folder `dinov3_gan/dinov3_weights` (please make the folder).

### 3. Prepare your training datasets

Start to train OMGSR-S-512:
```
bash train_omgsr_s_512.sh
```

Start to train OMGSR-F-1024:
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