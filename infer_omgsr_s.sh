python infer/infer_omgsr_s.py \
    --input_image my_tests \
    --output_dir experiments_omgsr_f \
    --sd_path stabilityai/stable-diffusion-2-1-base \
    --lora_path adapters/omgsr-s-512-adapter \
    --process_size 512 \
    --upscale 4 \
    --mid_timestep 273 \
    --align_method adain
