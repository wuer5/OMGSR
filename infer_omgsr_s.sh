# python infer/infer_omgsr_s.py \
#     --input_image /DataDisk1/wzq/projects/OMGSR/test_datasets/RealLQ250 \
#     --output_dir experiments_omgsr_s/RealLQ250_7000 \
#     --sd_path /DataDisk1/wzq/aigc_models/stable-diffusion-2-1-base \
#     --lora_path /DataDisk1/wzq/projects/OMGSR/trainings/omgsr_s_512/weight-7000 \
#     --process_size 512 \
#     --upscale 4 \
#     --mid_timestep 273 \
#     --prompt "" \
#     --align_method adain

# !/bin/bash

for i in $(seq 4000 100 6000); do
    echo "Testing weight-$i"
    python infer/infer_omgsr_s.py \
        --input_image /DataDisk1/wzq/projects/OMGSR/test_datasets/RealLQ250 \
        --output_dir experiments_omgsr_s/omgsr_s_512_mae/RealLQ250_${i} \
        --sd_path /DataDisk1/wzq/aigc_models/stable-diffusion-2-1-base \
        --lora_path /DataDisk1/wzq/projects/OMGSR/trainings/omgsr_s_512_mae/weight-${i} \
        --process_size 512 \
        --upscale 4 \
        --mid_timestep 273 \
        --prompt "" \
        --align_method adain
done