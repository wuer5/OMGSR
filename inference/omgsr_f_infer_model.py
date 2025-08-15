from typing import Callable
import torch
import os
import time
import torch
import torch
from peft import PeftModel
from diffusers import AutoencoderKL, FluxTransformer2DModel
from vaehook import VAEHook
import math
import numpy as np

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, _, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def get_flux_setting_timesteps(n=999):
    return get_schedule(
        n,
        (1024 // 8) * (1024 // 8) // 4,
        shift=True,
    )

def adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1024):
    w, h = image_size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        tile_w = min(w, max_tile_size)
        tile_h = min(int(tile_w / aspect_ratio), max_tile_size)
    else:
        tile_h = min(h, max_tile_size)
        tile_w = min(int(tile_h * aspect_ratio), max_tile_size)
    return max(tile_w, base_tile_size), max(tile_h, base_tile_size)

def create_gaussian_weight(tile_size, sigma=0.3):
    x = np.linspace(-1, 1, tile_size)
    y = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, y)
    gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

class OMGSR_F_Infer(torch.nn.Module):
    def __init__(self, flux_path, lora_path, device, weight_dtype=torch.bfloat16, mid_timestep=295, guidance_scale=1.0):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
            flux_path,
            subfolder="vae",
        )
        flux_transformer = FluxTransformer2DModel.from_pretrained(
            flux_path,
            subfolder="transformer",
        )

        vae.requires_grad_(False)
        flux_transformer.requires_grad_(False)

        vae.to(device=device, dtype=weight_dtype)  
        flux_transformer.to(dtype=weight_dtype, device=device)

        print("Loading adapers...")
        flux_transformer = PeftModel.from_pretrained(flux_transformer, os.path.join(lora_path, "flux_adapter"), is_trainable=False)
        vae.encoder = PeftModel.from_pretrained(vae.encoder, os.path.join(lora_path, "vae_encoder_adapter"), is_trainable=False)
        flux_transformer.merge_and_unload()
        vae.encoder.merge_and_unload()
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.guidance_scale = guidance_scale
        self.mid_timestep = mid_timestep
        self.weight_dtype = weight_dtype

        flux_timesteps = get_flux_setting_timesteps()
        self.t_curr = flux_timesteps[-(self.mid_timestep + 1)]
        self.t_prev = flux_timesteps[-1]   # 0.0
        print(f"Current One mid-timestep settings: {mid_timestep}")

        self.vae = vae
        self.flux_transformer = flux_transformer
        self.device = device
        self._init_tiled_vae(encoder_tile_size=1024, decoder_tile_size=224)

    def _init_tiled_vae(
            self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, 16, 1, 1))

    def _forward_no_tile(self, lq_latent, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids):
        bsz, c, h, w = lq_latent.shape
        guidance_vec = torch.full(
            (bsz,),
            self.guidance_scale,
            device=lq_latent.device,
            dtype=self.weight_dtype,
        )
       
        lq_latent = _pack_latents(
            lq_latent,
            batch_size=bsz,
            num_channels_latents=c,
            height=h,
            width=w,
        )
        # One-Step Predict.
        model_pred = self.flux_transformer(
            hidden_states=lq_latent,
            timestep=torch.tensor([self.t_curr], device=lq_latent.device),  # One-Step
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        lq_latent = lq_latent + (self.t_prev - self.t_curr) * model_pred  # 注意

        lq_latent = _unpack_latents(
            lq_latent,
            height=h * self.vae_scale_factor,
            width=w * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        lq_latent = (lq_latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_img = self.vae.decode(lq_latent.to(self.vae.dtype), return_dict=False)[0]
        return pred_img

    def _forward_tile(self, lq_latent, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, tile_size, tile_overlap):
        _, _, h, w = lq_latent.shape
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        grid_rows = 0
        cur_x = 0
        while cur_x < lq_latent.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < lq_latent.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    bsz, c, ht, wt = input_list_t.shape
                    # predict the noise residual
                    guidance_vec = torch.full(
                        (bsz,),
                        self.guidance_scale,
                        device=input_list_t.device,
                        dtype=self.flux_transformer.dtype,
                    )
                    input_list_t = _pack_latents(
                        input_list_t,
                        batch_size=bsz,
                        num_channels_latents=c,
                        height=ht,
                        width=wt,
                    )
                    # One-Step Predict.
                    model_out = self.flux_transformer(
                        hidden_states=input_list_t,
                        timestep=torch.tensor([self.t_curr], device=input_list_t.device),  # One-Step
                        guidance=guidance_vec,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    model_out = _unpack_latents(
                        model_out,
                        height=ht * self.vae_scale_factor,
                        width=wt * self.vae_scale_factor,
                        vae_scale_factor=self.vae_scale_factor,
                    )
                    input_list = []
                noise_preds.append(model_out)
        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
        contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size
                # print("----", noise_preds[row*grid_cols + col].shape, tile_weights.shape)
                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        lq_latent = lq_latent + (self.t_prev - self.t_curr) * model_pred  
        lq_latent = (lq_latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_img = self.vae.decode(lq_latent.to(self.vae.dtype), return_dict=False)[0]

        return pred_img

    def forward(self, lq_img, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, tile_size, tile_overlap):
        # torch.cuda.synchronize()
        # start_time = time.time()
        lq_latent = encode_images(
            lq_img.to(self.vae.dtype), self.vae, self.weight_dtype
        )
        _, _, h, w = lq_latent.shape
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            pred_img = self._forward_no_tile(lq_latent, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids)
        else:
            print(f"[Tiled Latent]: the input size is {lq_img.shape[-2]}x{lq_img.shape[-1]}, need to tiled")
            pred_img = self._forward_tile(lq_latent, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, tile_size, tile_overlap)
        # torch.cuda.synchronize()
        # total_time = time.time() - start_time
        # print(total_time)
        return pred_img
    