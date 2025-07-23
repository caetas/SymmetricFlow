# Training the Models

You can launch a training session from the command line or by editing [`main.sh`](./../src/symmetricflow/scripts/main.sh) and using Docker or Apptainer.

## Multi-GPU support and Mixed Precision

All models in this repository support Multi-GPU and Mixed Precision training.

[`Accelerate`](https://huggingface.co/docs/accelerate/en/index) should be configured for your hardware setup using:

    accelerate config

To use these features, the models that should be launched with:

    accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...

## Stable Diffusion Models Training

You can find out more about the parameters by checking [`util.py`](./../src/symmetricflow/utils/util.py) or by running the following command on the example script:

    python mainsd.py --help

### CelebAMask-HQ

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

### COCO-Stuff

    accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 mainsd.py \
        --train \
        --dataset coco \
        --beta 6.0 \
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

## Classificication Models Training

You can find out more about the parameters by checking [`util.py`](./../src/symmetricflow/utils/util.py) or by running the following command on the example script:

    python classification.py --help

### MNIST

    accelerate launch classification.py \
        --train \
        --dataset mnist \
        --snapshot 10 \
        --n_epochs 1000 \
        --lr 5e-4 \
        --warmup 100 \
        --decay 0.0 \
        --sample_and_save_freq 20 \
        --num_workers 8 \
        --model_channels 32 \
        --num_res_blocks 2 \
        --channel_mult 1 2 2 2 \
        --num_heads 4 \
        --num_head_channels 64 \
        --attention_resolutions 16 \
        --batch_size 512 \
        --solver_lib torchdiffeq \
        --solver euler \
        --step_size 0.04 \
        --beta 4


### CIFAR-10

    accelerate launch classification.py \
        --train \
        --dataset cifar10 \
        --snapshot 10 \
        --n_epochs 1000 \
        --lr 3e-4 \
        --warmup 100 \
        --decay 0.0 \
        --sample_and_save_freq 20 \
        --num_workers 16 \
        --model_channels 256 \
        --num_res_blocks 2 \
        --channel_mult 1 2 2 2 \
        --num_heads 4 \
        --num_head_channels 64 \
        --attention_resolutions 16 \
        --batch_size 180 \
        --solver_lib torchdiffeq \
        --solver euler \
        --step_size 0.04 \
        --beta 4