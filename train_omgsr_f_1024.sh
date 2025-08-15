CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --main_process_port 29500 \
    train_omgsr_f.py --config ./configs/omgsr_f_1024.yml