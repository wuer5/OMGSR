CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --main_process_port 29500 \
    train_omgsr_s.py --config ./configs/omgsr_s_512.yml