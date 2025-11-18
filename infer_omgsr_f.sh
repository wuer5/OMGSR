python infer/infer_omgsr_f.py \
    --input_image my_tests \
    --output_dir experiments_omgsr_f \
    --flux_path black-forest-labs/FLUX.1-dev \
    --lora_path adapters/omgsr-f-1024-adapter \
    --process_size 1024 \
    --upscale 4 \
    --mid_timestep 244 \
    --align_method adain



