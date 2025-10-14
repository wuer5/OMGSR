#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys
import argparse
import logging
import math
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
import warnings
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.my_dataset import PairedDataset
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
import copy

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = pixel_latents* vae.config.scaling_factor
    return pixel_latents

def set_vae_encoder_lora(vae_encoder, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vae_encoder = PeftModel(
        model=vae_encoder,
        peft_config=vae_encoder_lora_config,
        adapter_name="vae_encoder_lora_adapter",
    )
    vae_encoder.print_trainable_parameters()
    return vae_encoder

def set_unet_lora(unet, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    unet = PeftModel(
        model=unet,
        peft_config=unet_lora_config,
        adapter_name="unet_lora_adapter",
    )
    unet.print_trainable_parameters()
    return unet

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/omgsr_s_512.yml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    logger.info(f"Using mixed_precision: {args.mixed_precision}")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))

    if not args.fixed_prompt_path:
        logger.info(f"Current prompt: {args.fixed_prompt}")
        tokenizer = AutoTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            args.sd_path, subfolder="text_encoder"
        ).to(accelerator.device)

        def encode_prompt(prompt_batch):
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
                        ).input_ids.to(text_encoder.device)
                    )[0]
                    for caption in prompt_batch
                ]
            return torch.concat(prompt_embeds, dim=0)
        prompt_embeds = encode_prompt([args.fixed_prompt] * args.train_batch_size)
    else:
        prompt_embeds = torch.load(args.fixed_prompt_path, map_location=accelerator.device)

    # mid-timestep
    mid_timestep = args.mid_timestep
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    sqrt_alphas_cumprod_t = math.sqrt(
        noise_scheduler.alphas_cumprod[mid_timestep]
    )
    sqrt_one_minus_alphas_cumprod_t = math.sqrt(
        1 - noise_scheduler.alphas_cumprod[mid_timestep]
    )
    logger.info(f"Current {args.model} mid-timestep = {mid_timestep}")
    # vae
    fixed_vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae")
    lora_vae = copy.deepcopy(fixed_vae)
    del lora_vae.decoder
    free_memory()
    fixed_vae.requires_grad_(False)
    lora_vae.requires_grad_(False)
    lora_vae.encoder = set_vae_encoder_lora(lora_vae.encoder, args.vae_lora_rank)
    # unet
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet")
    unet.requires_grad_(False)
    unet = set_unet_lora(unet, args.unet_lora_rank)
    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )
        
    # DINOv3-ConvNeXt DISTS Loss
    from dinov3_gan.dinov3_convnext_dists import DINOv3ConvNeXtDISTS
    dists_fn = DINOv3ConvNeXtDISTS(dinov3_convnext_size=args.dinov3_convnext_size)

    # DINOv3-ConvNeXt Discrminator
    from dinov3_gan.dinov3_convnext_disc import Dinov3ConvNeXtDiscriminator
    net_disc = Dinov3ConvNeXtDiscriminator(dinov3_convnext_size=args.dinov3_convnext_size, resolution=args.resolution)

    fixed_vae.to(accelerator.device)  
    lora_vae.to(accelerator.device) 
    unet.to(device=accelerator.device)
    net_disc.to(accelerator.device)
    dists_fn.to(accelerator.device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        f"Total vae_encoder training parameters: {sum([p.numel() for p in lora_vae.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total unet training parameters: {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total disc training parameters: {sum([p.numel() for p in net_disc.parameters() if p.requires_grad]) / 1000000} M"
    )
    sr_opt = list(filter(lambda p: p.requires_grad, lora_vae.parameters())) + list(filter(lambda p: p.requires_grad, unet.parameters()))
    disc_opt = list(filter(lambda p: p.requires_grad, net_disc.parameters()))

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_sr = optimizer_class(
        sr_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_disc = optimizer_class(
        disc_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PairedDataset(args.dataset_txt_or_dir_paths, args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        lora_vae,
        unet,
        net_disc,
        optimizer_sr,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_sr,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        lora_vae,
        unet,
        net_disc,
        optimizer_sr,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_sr,
        lr_scheduler_disc,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info(f"***** Start training {args.model} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            # global_step = int(path.split("-")[1])

            # initial_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch
            # TODO
            pass

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    def one_mid_timestep_pred(lq_latent):
        model_pred = unet(lq_latent, mid_timestep, encoder_hidden_states=prompt_embeds).sample
        denoised_latent = (lq_latent - sqrt_one_minus_alphas_cumprod_t * model_pred) / sqrt_alphas_cumprod_t
        pred_img = (fixed_vae.decode(denoised_latent / fixed_vae.config.scaling_factor).sample).clamp(-1, 1)
        return pred_img
    
    for epoch in range(first_epoch, args.num_train_epochs):
        lora_vae.train()
        unet.train()
        net_disc.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*[lora_vae, unet, net_disc]):
                # Prepare data
                lq_img, hq_img = batch
                lq_img = lq_img.to(accelerator.device)
                hq_img = hq_img.to(accelerator.device)
                with torch.no_grad():
                    hq_latent = encode_images(hq_img, fixed_vae)
                    noise = torch.randn_like(hq_latent)
                    pretrained_noisy_latent = sqrt_alphas_cumprod_t * hq_latent + sqrt_one_minus_alphas_cumprod_t * noise 

                lq_latent = encode_images(lq_img, unwrap_model(lora_vae))

                # LRR Loss: Latent Representation Refinement Loss
                loss_LRR = F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean") * args.lambda_LRR
                
                # Onestep prediction at mid-timestep
                pred_img = one_mid_timestep_pred(lq_latent)

                # DINOv3-ConvNext DISTS Loss
                loss_Dv3D = dists_fn(pred_img, hq_img).mean() * args.lambda_Dv3D

                # L1 Loss
                loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * args.lambda_L1

                # Generator Loss (SD/FLUX)
                loss_G = net_disc(pred_img, for_G=True).mean() * args.lambda_GAN
                
                total_G_loss = loss_LRR + loss_Dv3D + loss_G + loss_L1

                accelerator.backward(total_G_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_opt, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad()
                
                fake_img = pred_img.detach()
                # Fake images
                loss_D_fake = net_disc(fake_img, for_real=False).mean() * args.lambda_GAN 
                # Real images
                loss_D_real = net_disc(hq_img, for_real=True).mean() * args.lambda_GAN 
          
                total_D_loss = loss_D_real + loss_D_fake 

                accelerator.backward(total_D_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_opt, args.max_grad_norm)

                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (
                    accelerator.is_main_process
                    and global_step % args.save_img_steps == 0
                ):
                    img_path = os.path.join(args.output_dir, f"img-{global_step}.jpg")
                    save_imgs = (torch.stack([lq_img[0], pred_img[0], hq_img[0]], dim=0) + 1) / 2
                    save_image(save_imgs.detach(), img_path)
                    logger.info(f"img-{global_step}.jpg saved!")

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    # if global_step % args.checkpointing_steps == 0:
                    if global_step in list(range(4000, 6001, 100)):

                        weight_path = os.path.join(
                            args.output_dir, f"weight-{global_step}"
                        )
                        os.makedirs(weight_path, exist_ok=True)
                        unwrap_model(unet).save_pretrained(weight_path)
                        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss_LRR": loss_LRR.detach().item(),
                "loss_D_fake": loss_D_fake.detach().item(),
                "loss_D_real": loss_D_real.detach().item(),
                "loss_Dv3D": loss_Dv3D.detach().item(),
                "loss_L1": loss_L1.detach().item(),
                "lr": lr_scheduler_sr.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        unwrap_model(unet).save_pretrained(weight_path)
        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
