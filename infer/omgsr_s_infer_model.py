import os
import torch
import time
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel
from .vaehook import VAEHook

class OMGSR_S_Infer(torch.nn.Module):
    def __init__(self, sd_path, lora_path, mid_timestep, device, weight_dtype):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.mid_timestep = mid_timestep
        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.alpha_t = self.scheduler.alphas_cumprod[mid_timestep]
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        vae.encoder = PeftModel.from_pretrained(
            vae.encoder, os.path.join(lora_path, "vae_encoder_lora_adapter")
        )
        unet = PeftModel.from_pretrained(
            unet, os.path.join(lora_path, "unet_lora_adapter")
        )
        vae.encoder.merge_and_unload()
        unet.merge_and_unload()
        vae = vae.to(device=device)
        unet = unet.to(device=device, dtype=weight_dtype)
        print(f"Current One mid-timestep settings: {mid_timestep}")
        self.vae = vae
        self.unet = unet
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
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


    def _forward_no_tile(self, lq_latent, prompt_embeds):
        model_pred = self.unet(
            lq_latent.to(self.unet.dtype),
            self.mid_timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        denoised_latent = (
            lq_latent - (1 - self.alpha_t).sqrt() * model_pred.to(self.vae.dtype)
        ) / self.alpha_t.sqrt()
        pred_img = (
            self.vae.decode(denoised_latent / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return pred_img

    def _forward_tile(self, lq_latent, prompt_embeds, tile_size, tile_overlap):
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
                    # predict the noise residual
                    model_out = self.unet(input_list_t.to(self.unet.dtype), self.mid_timestep, encoder_hidden_states=prompt_embeds.to(self.unet.dtype),).sample
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

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        lq_latent = (
            lq_latent - (1 - self.alpha_t).sqrt() * model_pred.to(self.vae.dtype)
        ) / self.alpha_t.sqrt()
        pred_image = (self.vae.decode(lq_latent / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return pred_image


    def forward(self, lq_img, prompt_embeds, tile_size, tile_overlap):
        lq_latent = self.vae.encode(lq_img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        ## add tile function
        _, _, h, w = lq_latent.size()
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            pred_img = self._forward_no_tile(lq_latent, prompt_embeds)
        else:
            print(f"[Tiled Latent]: the input size is {lq_img.shape[-2]}x{lq_img.shape[-1]}, need to tiled")
            pred_img = self._forward_tile(lq_latent, prompt_embeds, tile_size, tile_overlap)
        return pred_img
