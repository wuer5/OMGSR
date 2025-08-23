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
from infer.omgsr_f_infer_model import OMGSR_F_Infer
from diffusers.training_utils import free_memory
from diffusers import FluxPipeline

def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def main(args):
    # Initialize the model
    omgsr = OMGSR_F_Infer(
        args.flux_path, args.lora_path, device=args.device, 
        guidance_scale=args.guidance_scale, mid_timestep=args.mid_timestep)

    text_encoding_pipeline = FluxPipeline.from_pretrained(
        args.flux_path, transformer=None, vae=None, torch_dtype=args.weight_dtype
    )
    text_encoding_pipeline = text_encoding_pipeline.to("cuda")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            args.prompt, prompt_2=None
        )

    # Release the pipeline
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")  # Move to CPU first
    del text_encoding_pipeline
    free_memory()

    latent_image_ids = _prepare_latent_image_ids(
        (args.process_size // 8) // 2,
        (args.process_size // 8) // 2,
        args.device,
        args.weight_dtype,
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

    for image_name in tqdm(image_names):
        # Load the input image
        input_image = Image.open(image_name).convert("RGB")
        ori_width, ori_height = input_image.size
        new_width, new_height = ori_width * args.upscale, ori_height * args.upscale

        # Upscale the image
        input_image = input_image.resize((new_width, new_height))

        # Pad the image to make it a multiple of process_size
        def pad_to_multiple(image, multiple):
            width, height = image.size
            pad_width = (multiple - (width % multiple)) % multiple
            pad_height = (multiple - (height % multiple)) % multiple
            
            # Only pad if necessary
            if pad_width > 0 or pad_height > 0:
                # Create a new image with the padded size
                padded_image = Image.new("RGB", 
                                    (width + pad_width, height + pad_height),
                                    (0, 0, 0))  # Black padding
                padded_image.paste(image, (0, 0))  # Paste original at top-left
                return padded_image, (width, height)  # Return padded image and original size
            return image, (width, height)

        # Pad to multiple of process_size
        padded_image, original_padded_size = pad_to_multiple(input_image, args.process_size)
        padded_width, padded_height = padded_image.size
        print(f"Process size: {padded_width}x{padded_height}")
        bname = os.path.basename(image_name).split('.')[0] + ".png"
        
        tile_size = args.process_size // 8
        tile_overlap = tile_size // 4
        # Process the image
        with torch.no_grad():
            lq_img = F.to_tensor(padded_image).unsqueeze(0).to(device=args.device, dtype=args.weight_dtype) * 2 - 1
            output_image = omgsr(lq_img.to(), prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, tile_size, tile_overlap)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].float().cpu())

        # Crop back to the original padded size before upscaling
        output_pil = output_pil.crop((0, 0, original_padded_size[0], original_padded_size[1]))

        # Apply color correction if needed
        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)
        
        output_pil.save(os.path.join(args.output_dir, bname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMGSR-F Inference Script")
    
    # Required arguments
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image or directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory for output results")
    parser.add_argument("--flux_path", type=str, required=True,
                        help="Path to FLUX model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA model")
    
    # Optional arguments with defaults
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device")
    parser.add_argument("--process_size", type=int, default=1024,
                        help="Processing size for the model")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscaling factor")
    parser.add_argument("--align_method", type=str, default="adain",
                        choices=["wavelet", "adain", "nofix"],
                        help="Color alignment method")
    parser.add_argument("--weight_dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Weight data type")
    parser.add_argument("--prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
                        help="Prompt for text conditioning")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for generation")
    parser.add_argument("--mid_timestep", type=int, default=295,
                        help="Mid timestep for generation")
    
    args = parser.parse_args()
    
    # Convert weight_dtype string to torch dtype
    args.weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }[args.weight_dtype]
    
    main(args)
