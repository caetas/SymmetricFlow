# Inference and Evaluation

## Sampling

You can sample from a pretrained model using:

    accelerate launch --mixed_precision=bf16 mainsd.py \
        --sample \
        --dataset coco \
        --beta 6.0 \
        --latent \
        --size 512 \
        --num_samples 4 \
        --solver_lib torchdiffeq \
        --solver euler \
        --step_size 0.04 \
        --checkpoint ../../models/SymmetricalFlowMatchingSD/LatFM_coco_beta6.0.pt

## Evaluate Segmentation

You can evaluate a pretrained model using:

    accelerate launch --mixed_precision=bf16 mainsd.py \
        --eval \
        --dataset coco \
        --beta 6.0 \
        --latent \
        --size 512 \
        --batch_size 32 \
        --solver_lib torchdiffeq \
        --solver euler \
        --step_size 0.04 \
        --checkpoint ../../models/SymmetricalFlowMatchingSD/LatFM_coco_beta6.0.pt

## Classification

If you want to classify images from the test set, you can use:

    accelerate launch classification.py \
        --classification \
        --dataset mnist \
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
        --step_size 1 \
        --beta 4 \
        --checkpoint ../../models/SymmetricalFlowMatchingClass/FM_mnist_beta4.0.pt

## FID

To generate the images to evaluate FID please run:

    accelerate launch --mixed_precision=bf16 mainsd.py \
        --fid \
        --dataset coco \
        --beta 6.0 \
        --latent \
        --size 512 \
        --batch_size 2 \
        --solver_lib torchdiffeq \
        --solver euler \
        --step_size 0.04 \
        --checkpoint ../../models/SymmetricalFlowMatchingSD/LatFM_coco_beta6.0.pt

Use `pytorch-fid` to then calculate the FID score.

