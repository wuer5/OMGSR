python infer_omgsr_f.py \
    --input_image tests \
    --output_dir experiments_omgsr_f \
    --flux_path black-forest-labs/FLUX.1-dev \
    --lora_path adapters/omgsr-f-1024-weight-8000 \
    --process_size 1024 \
    --upscale 4 \
    --align_method adain 



