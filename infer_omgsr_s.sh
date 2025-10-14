python infer/infer_omgsr_s.py \
    --input_image my_tests \
    --output_dir experiments_omgsr_f \
    --sd_path stabilityai/stable-diffusion-2-1 \
    --lora_path adapters/omgsr-f-1024-weight \
    --process_size 512 \
    --upscale 4 \
    --align_method adain
