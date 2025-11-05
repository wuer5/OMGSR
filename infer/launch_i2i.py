import os
import time
import torch
import ray
import io
import logging
import time
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from torchvision import transforms
import argparse
from PIL import Image
import torchvision.transforms.functional as F
from omgsr_f_infer_model import OMGSR_F_Infer
from infer.wavelet_color_fix import adain_color_fix, wavelet_color_fix

args = None
from diffusers import FluxPipeline
from infer_omgsr_f import _prepare_latent_image_ids


# Define request model
class GenerateRequest(BaseModel):
    resize_height: Optional[int] = Field(default=1224, description="图像缩放后高度")
    resize_width: Optional[int] = Field(default=1024, description="图像缩放后宽度")
    input_image: Optional[str] = Field(description="输入图像本地路径（如果不传image Base64，可传本地路径）")
    output_dir: Optional[str] = Field(default="experiments_omgsr_f", description="生成图像保存目录（服务器端）")
    process_size: Optional[int] = Field(default=1024, description="图像处理尺寸")
    upscale: Optional[int] = Field(default=4, description="输出图像放大倍数（1-8）")
    mid_timestep: Optional[int] = Field(default=244, description="超分中间时间步")
    align_method: Optional[str] = Field(default="adain", description="颜色对齐方式（二选一）")
    guidance_scale: Optional[float] = Field(default=1.0, description="prompt引导尺度")

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "resize_height": 1224,
                "resize_width": 1024,
                "input_image": "/data/input/test.jpg",
                "output_dir": "experiments_omgsr_f",
                "process_size": 1024,
                "upscale": 4,
                "mid_timestep": 244,
                "align_method": "adain",
                "guidance_scale": 1.0
            }
        }


app = FastAPI()


@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(self, rank: int, world_size: int, args):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        self.rank = rank
        self.args = args
        self.setup_logger()
        self._load_models()

    def _load_models(self):
        self.logger.info("Start loading models...")
        start_time = time.time()
        self.device = 'cuda:0'
        self.weight_dtype = torch.bfloat16
        self.omgsr = OMGSR_F_Infer(
            self.args.flux_path,  # 从全局 args 取模型路径
            self.args.lora_path,  # 从全局 args 取 LoRA 路径
            device=self.device,
            guidance_scale=1.0,  # 引导尺度可后续从请求体覆盖
            mid_timestep=244  # 中间时间步可后续从请求体覆盖
        )

        # 3. 加载文本编码流水线（FluxPipeline）
        self.text_encoding_pipeline = FluxPipeline.from_pretrained(
            self.args.flux_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype
        ).to(self.device)

        self.latent_image_ids = _prepare_latent_image_ids(
            (1024 // 8) // 2,
            (1024 // 8) // 2,
            self.device,
            self.weight_dtype
        )
        with torch.no_grad():
            self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = self.text_encoding_pipeline.encode_prompt(
                '', prompt_2=None
            )
        load_time = time.time() - start_time
        self.logger.info(f"Models loaded successfully! Cost time: {load_time:.2f} sec")

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def generate(self, request: GenerateRequest):
        try:
            total_time = 0
            with torch.no_grad():
                input_image = Image.open(request.input_image).convert('RGB')
                ori_width, ori_height = input_image.size
                print('Original input_image size:', input_image.size)
                rscale = request.upscale
                resize_flag = False
                args.process_size = 1024
                if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
                    scale = (args.process_size // rscale) / min(ori_width, ori_height)
                    input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
                    resize_flag = True

                input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
                new_width = input_image.width - input_image.width % 8
                new_height = input_image.height - input_image.height % 8
                input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
                bname = os.path.basename(request.input_image).split('/')[-1].split('.')[0] + ".png"
                tile_size = args.process_size // 8
                tile_overlap = tile_size // 4
                start_time = time.time()
                input_image = input_image.resize((request.resize_width, request.resize_height))
                print('input_image size after resize:', input_image.size)
                lq_img = F.to_tensor(input_image).unsqueeze(0).to(device='cuda:0', dtype=torch.bfloat16) * 2 - 1
                output_image, time_d = self.omgsr(lq_img, self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids,
                                                  self.latent_image_ids, tile_size, tile_overlap)
                cost_time = time.time() - start_time
                print(f"From client time cost: {cost_time:.2f} seconds")
                total_time += time_d

            output_image = output_image * 0.5 + 0.5
            output_image = torch.clip(output_image, 0, 1)
            output_pil = transforms.ToPILImage()(output_image[0].cpu().float())
            args.align_method = 'adain'
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize((int(request.upscale * ori_width), int(request.upscale * ori_height)))
            output_pil.save(os.path.join(request.output_dir, bname))
            cost_time = time.time() - start_time
            return {
                "message": "Image generated successfully",
                "elapsed_time": f"{cost_time:.2f} sec",
                "output": request.output_dir,
                "save_to_disk": True
            }
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


class Engine:
    def __init__(self, world_size: int):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()

        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(rank=rank, world_size=world_size, args=args)
            for rank in range(num_workers)
        ]

    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None)


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # Add input validation
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")

        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')

    parser.add_argument('--flux_path', type=str, help='model_path')
    parser.add_argument('--lora_path', type=str, help='lora_path')
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    # parser.add_argument('--resize_height', type=str, help='resize_height', required=True)
    # parser.add_argument('--resize_width', type=str, default='output', help='resize_width', required=True)
    # parser.add_argument('--input_image', type=str, default='my_tests', help='Path to input image')
    # parser.add_argument('--output_dir', type=str, default='experiments_omgsr_f', help='Path to save generated images')
    # parser.add_argument('--prcocess_size', type=int, default=1024, help='Process size for images')
    # parser.add_argument('--upscale', type=int, default=4, help='Upscale factor for output image')
    # parser.add_argument('--mid_timestep', type=int, default=244)
    # parser.add_argument('--align_method', type=str, default='adain', help='Color alignment method')

    args = parser.parse_args()

    engine = Engine(
        world_size=args.world_size,
    )

    # Start the server
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8188)