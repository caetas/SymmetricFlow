cd src/symmetricflow

accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 mainsd.py \
    --train \
    --dataset celeba \
    --beta 10.0 \
    --snapshot 10 \
    --latent \
    --size 512 \
    --n_epochs 200 \
    --lr 8e-5 \
    --warmup 10 \
    --decay 0.0 \
    --image_weight 0.7 \
    --sample_and_save_freq 10 \
    --num_workers 32 \
    --ema_rate 0.9 \
    --batch_size 32 \
    --solver_lib torchdiffeq \
    --solver euler \
    --step_size 0.1