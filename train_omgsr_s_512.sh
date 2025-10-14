export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --main_process_port 29500 \
    train/train_omgsr_s.py --config ./configs/omgsr_s_512.yml