python infer_omgsr_s.py \
    --input_image tests \
    --output_dir experiments_omgsr_s \
    --sd_path stabilityai/sd-turbo \
    --lora_path adapters/omgsr-s-512-weight-33000 \
    --process_size 512 \
    --upscale 2 \
    --align_method adain 
	