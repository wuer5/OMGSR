import sys
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from infer.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import glob
from infer.omgsr_s_infer_model import OMGSR_S_Infer
from diffusers.training_utils import free_memory
from transformers import AutoTokenizer, CLIPTextModel

def main(args):
    # Initialize the model
    tokenizer = AutoTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_path, subfolder="text_encoder"
    ).to(args.device, dtype=args.weight_dtype)
    
    def encode_prompt(prompt_batch):
        print(f"Current prompt: {prompt_batch}")
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                text_encoder(
                    tokenizer(
                        caption,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device=text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    prompt_embeds = encode_prompt([args.prompt])
    del tokenizer
    del text_encoder
    free_memory()
    net_sr = OMGSR_S_Infer(
        sd_path=args.sd_path,
        lora_path=args.lora_path,
        mid_timestep=args.mid_timestep,
        device=args.device,
        weight_dtype=args.weight_dtype
    )
    
    if ".txt" in args.input_image:
        with open(args.input_image, 'r') as f:
            image_names = [l.strip() for l in f.readlines()]
    else:
        # Get all input images
        if os.path.isdir(args.input_image):
            image_names = sorted(glob.glob(f"{args.input_image}/*.png") + glob.glob(f"{args.input_image}/*.jpg") + glob.glob(f"{args.input_image}/*.jpeg"))
        else:
            image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"There are {len(image_names)} images.")

    total_time = 0
    for image_name in tqdm(image_names):
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name).split('.')[0] + ".png"
        
        tile_size = args.process_size // 8
        tile_overlap = tile_size // 2

        # Process the image
        with torch.no_grad():
            lq_img = F.to_tensor(input_image).unsqueeze(0).to(device=args.device, dtype=args.weight_dtype) * 2 - 1
            output_image, time = net_sr(lq_img, prompt_embeds, tile_size, tile_overlap)
            total_time += time

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1).float()
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    print(f"Average inference time: {total_time / len(image_names)}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMGSR-S Inference Script")
    
    # Required arguments
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image or directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--sd_path", type=str, required=True,
                        help="Path to Stable Diffusion model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA model")
    
    # Optional arguments with defaults
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device")
    parser.add_argument("--process_size", type=int, default=512,
                        help="Processing size for the model")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscaling factor")
    parser.add_argument("--align_method", type=str, default="adain",
                        choices=["wavelet", "adain", "nofix"],
                        help="Color alignment method")
    parser.add_argument("--weight_dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Weight data type")
    parser.add_argument("--prompt", type=str, nargs="+",
                        default="",
                        help="Prompt for text conditioning (can specify multiple)")
    parser.add_argument("--mid_timestep", type=int, default=273,
                        help="Mid timestep for generation")
    args = parser.parse_args()
    
    # Convert weight_dtype string to torch dtype
    args.weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }[args.weight_dtype]
    
    main(args)
