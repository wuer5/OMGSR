import math
import torch
from tqdm import tqdm
import argparse
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.my_dataset import PairedDataset
from diffusers import AutoencoderKL, DDPMScheduler

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = pixel_latents * vae.config.scaling_factor
    return pixel_latents

def main():
    args = parse_args()
    
    device = args.device
    select_timestep = list(range(999, -1, -1))
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    vae.requires_grad_(False)
    
    loss_accumulators = {t: 0.0 for t in select_timestep}
    sample_counts = {t: 0 for t in select_timestep}
    
    train_dataset = PairedDataset(args.dataset_txt_or_dir_paths, args.resolution)
    
    if args.max_samples is not None and args.max_samples < len(train_dataset):
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(min(args.max_samples, len(train_dataset)))
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    pbar = tqdm(train_dataloader, desc="Finding optimal mid-timestep")
   
    for batch_idx, batch in enumerate(pbar):
        lq, hq = batch
        lq = lq.to(device)
        hq = hq.to(device)
        
        with torch.no_grad():
            lq_latent = encode_images(lq, vae)
            hq_latent = encode_images(hq, vae)

        batch_size = hq.shape[0]
        
        batch_loss_results = {}
        for t in select_timestep:
            sqrt_alphas_cumprod_t = math.sqrt(
                noise_scheduler.alphas_cumprod[t]
            )
            sqrt_one_minus_alphas_cumprod_t = math.sqrt(
                1 - noise_scheduler.alphas_cumprod[t]
            )

            signal_power = torch.mean((hq_latent) ** 2) * (sqrt_alphas_cumprod_t**2)
            noise_power  = sqrt_one_minus_alphas_cumprod_t ** 2
            SNR1 = signal_power / noise_power

            signal_power2 = torch.mean(hq_latent ** 2)
            noise_power2 = torch.mean((lq_latent - hq_latent) ** 2)
            SNR2 = signal_power2 / noise_power2
            loss = torch.abs(SNR1 - SNR2)

            loss_accumulators[t] += loss.item() * batch_size
            sample_counts[t] += batch_size
            
            batch_loss_results[t] = loss.item()
        
        current_avg_loss = {t: loss_accumulators[t] / max(sample_counts[t], 1) for t in select_timestep}
        
        best_t = min(current_avg_loss, key=current_avg_loss.get)
        best_loss = current_avg_loss[best_t]
        
        pbar.set_postfix({
            'best_sqrt_alphas_cumprod_t': math.sqrt(
                noise_scheduler.alphas_cumprod[best_t]
            ),
            'best_sqrt_one_minus_alphas_cumprod_t': math.sqrt(
                1 - noise_scheduler.alphas_cumprod[best_t]
            ),
            'best_t': best_t,
            'best_loss': f'{best_loss:.6f}',
            'batch': batch_idx + 1
        })

    final_avg_loss = {t: loss_accumulators[t] / max(sample_counts[t], 1) for t in select_timestep}
    optimal_t = min(final_avg_loss, key=final_avg_loss.get)

    print(f"Optimal timestep t: {optimal_t}")
    print(f"Loss: {final_avg_loss[optimal_t]:.6f}")

    import matplotlib.pyplot as plt

    best_t, best_loss = min(final_avg_loss.items(), key=lambda x: x[1])
    print(f"Optimal mid-timestep: t={best_t} with loss={best_loss:.6f}\n")

    plt.figure(figsize=(12, 7))
    timesteps = list(final_avg_loss.keys())
    losses = list(final_avg_loss.values())

    plt.plot(timesteps, losses, marker='o', linestyle='-', color='b', linewidth=1, markersize=6)

    plt.plot(best_t, best_loss, 'ro', markersize=12, markerfacecolor='none', 
            markeredgewidth=2, label=f'Optimal mid-timestep t={best_t} (loss={best_loss:.6f})')

    plt.annotate(f'Optimal: t={best_t}\nloss={best_loss:.6f}', 
                xy=(best_t, best_loss), 
                xytext=(best_t+5, best_loss+0.1),  
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.2))

    plt.title(f'Loss Over Timesteps (Optimal mid-timestep t={best_t})', fontsize=16)
    plt.xlabel('Timestep (t)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    plt.tight_layout()

    plt.savefig(f'Sd_{args.resolution}_optimal_mid-timestep.png', dpi=300, bbox_inches='tight')

    plt.show()

    print(f"Saved loss_with_optimal_t.png")
    print(f"Optimal mid-timestep t={best_t} (loss={best_loss:.6f})")

def parse_args():
    parser = argparse.ArgumentParser(description="Find optimal timestep for diffusion model")
    
    parser.add_argument("--dataset_txt_or_dir_paths", type=list, nargs='+', default=["/data/vjuicefs_ai_camera_aigc/public_data/11183307/omgsr-github-new/OMGSR-2/OMGSR/configs/LSDIR_paths.txt", "/data/vjuicefs_ai_camera_aigc/public_data/11183307/OMGSR-sd2.1-base/dataloaders/FFHQ10000_paths.txt"], help="List of dataset paths or txt files containing paths")
    parser.add_argument("--sd_path", default="/data/vjuicefs_ai_camera_aigc/public_data/aigc_models/models--stabilityai--stable-diffusion-2-1-base",
                       help="Path to sd model")
    parser.add_argument("--resolution", type=int, default=512, 
                       help="Image resolution")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of workers for dataloader")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()