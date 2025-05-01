from diffusers import AutoencoderKL
import torch
from data.Dataloaders import celeb_hq_masked_dataloader, cocostuff_dataloader
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import wandb
from config import models_dir
import os
from models.SymmFM import SymmFM

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=2, help='batch size')
    argparser.add_argument('--dataset', type=str, default='celeba', help='dataset name', choices=['celeba', 'coco'])
    argparser.add_argument('--size', type=int, default=256, help='image size')
    argparser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    argparser.add_argument('--n_steps', type=int, default=2000, help='number of steps for training')
    # model args
    argparser.add_argument('--pred', action='store_true', help='use prediction step')
    argparser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    argparser.add_argument('--model_channels', type=int, default = 64, help='number of features')
    argparser.add_argument('--num_res_blocks', type=int, default = 2, help='number of residual blocks per downsample')
    argparser.add_argument('--attention_resolutions', type=int, nargs='+', default = [4], help='downsample rates at which attention will take place')
    argparser.add_argument('--dropout', type=float, default = 0.0, help='dropout probability')
    argparser.add_argument('--channel_mult', type=int, nargs='+', default = [1, 2, 2], help='channel multiplier for each level of the UNet')
    argparser.add_argument('--conv_resample', type=bool, default = True, help='use learned convolutions for upsampling and downsampling')
    argparser.add_argument('--dims', type=int, default = 2, help='determines if the signal is 1D, 2D, or 3D')
    argparser.add_argument('--num_heads', type=int, default = 4, help='number of attention heads in each attention layer')
    argparser.add_argument('--num_head_channels', type=int, default = 32, help='use a fixed channel width per attention head')
    argparser.add_argument('--use_scale_shift_norm', type=bool, default = False, help='use a FiLM-like conditioning mechanism')
    argparser.add_argument('--resblock_updown', type=bool, default = False, help='use residual blocks for up/downsampling')
    argparser.add_argument('--use_new_attention_order', type=bool, default = False, help='use a different attention pattern for potentially increased efficiency')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--solver_lib', type=str, default='none', help='solver library', choices=['torchdiffeq', 'zuko', 'none'])
    argparser.add_argument('--step_size', type=float, default=0.1, help='step size for ODE solver')
    argparser.add_argument('--solver', type=str, default='dopri5', help='solver for ODE', choices=['dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun', 'midpoint', 'explicit_adams', 'implicit_adams'])
    argparser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    argparser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    argparser.add_argument('--decay', type=float, default=1e-7, help='decay rate')
    argparser.add_argument('--latent', action='store_true', default=False, help='Use latent implementation')
    argparser.add_argument('--ema_rate', type=float, default=0.999, help='ema rate')
    argparser.add_argument('--snapshots', type=int, default=10, help='how many snapshots during training')
    argparser.add_argument('--beta', type=float, default=10, help='Dequantization factor for the mask')
    argparser.add_argument('--image_weight', type=float, default=.9, help='Weight for the image loss')
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    return argparser.parse_args()

@torch.no_grad()
def validation_loss(vae, dataloader_val, criterion, device, pred=False, model=None):
    val_loss = 0.0
    vae.decoder.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloader_val, desc="Validation Batches", leave=False):
            masks = masks.to(device)
            # perturb input with uniform noise
            masks = masks + 10*(torch.randn_like(masks) * 0.5)/127.5

            # Obtain latents from frozen encoder
            latents = vae.encode(masks).latent_dist.sample()

            # Decode to reconstructed masks
            recon = vae.decode(latents).sample

            # Compute loss against ground-truth masks
            loss = criterion(recon, masks)

            if pred:
                # If using prediction step, compute the prediction loss
                if model is not None:
                    images = images.to(device)
                    images = vae.encode(images).latent_dist.sample().mul_(0.18215)
                    pred_masks = model.segment(images.shape[0], images, train=False, eval=True, fine_tune=True)
                    pred_masks = vae.decode(pred_masks/0.18215).sample
                    loss += criterion(pred_masks, masks)


            val_loss += loss.item()* masks.size(0)

    avg_val_loss = val_loss / len(dataloader_val.dataset)
    vae.decoder.train()
    return avg_val_loss

if __name__ == '__main__':

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    #vae.decoder.load_state_dict(torch.load(os.path.join(models_dir, "vae_decoder_epoch_step_0.pth")))

    if args.pred:
        model = SymmFM(args, args.size, 3)
        if args.checkpoint is not None:
            model.load_checkpoint(args.checkpoint)
        model.eval()
        model.to(device)

    else:
        model = None


    # Freeze encoder
    for param in vae.encoder.parameters():
        param.requires_grad = False
    vae.encoder.eval()

    # Decoder to train
    vae.decoder.train()

    # --------- Optimizer and Loss ---------
    optimizer = torch.optim.Adam(vae.decoder.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # --------- Training Loop ---------
    num_epochs = args.n_epochs

    if args.dataset == 'celeba':
        _,_,dataloader = celeb_hq_masked_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, input_shape=args.size, mode='train')
        _,_,dataloader_val = celeb_hq_masked_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, input_shape=args.size, mode='validation')
    elif args.dataset == 'coco':
        _,_,dataloader = cocostuff_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, input_shape=args.size, mode='train')
        _,_,dataloader_val = cocostuff_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, input_shape=args.size, mode='val')

    wandb.init(project="VAE-Finetuning",
                config={
                    "learning_rate": args.lr,
                    "architecture": "VAE",
                    "dataset": args.dataset,
                    "input_size": args.size,
                    "epochs": num_epochs,
                    "batch_size": args.batch_size,
                    "steps": args.n_steps,
                },
                name=f"VAE-Finetuning-{args.dataset}-{args.size}")
    cnt = 0
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs"):
        epoch_loss = 0.0
        for images,masks in tqdm(dataloader, desc="Training Batches", leave=False):
            masks = masks.to(device)
            # perturb input with uniform noise
            masks = masks + 10*(torch.randn_like(masks) * 0.5)/127.5

            # Obtain latents from frozen encoder
            with torch.no_grad():
                latents = vae.encode(masks).latent_dist.sample()

            # Decode to reconstructed masks
            recon = vae.decode(latents).sample

            # Compute loss against ground-truth masks
            loss = criterion(recon, masks)

            if args.pred:
                # If using prediction step, compute the prediction loss
                if model is not None:
                    images = images.to(device)
                    with torch.no_grad():
                        images = vae.encode(images).latent_dist.sample().mul_(0.18215)
                        pred_masks = model.segment(images.shape[0], images, train=False, eval=True, fine_tune=True)
                    pred_masks = vae.decode(pred_masks/0.18215).sample
                    loss += criterion(pred_masks, masks)
                    loss = loss*0.5

            # Backpropagate only through decoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()* masks.size(0)
            
            if cnt % 500 == 0:
                # Log validation loss
                val_loss = validation_loss(vae, dataloader_val, criterion, device)
                wandb.log({"validation_loss": val_loss})
                # Save model checkpoint
                recon = recon.cpu().detach().numpy()
                masks = masks.cpu().detach().numpy()
                recon = (recon * 127.5 + 127.5).clip(0, 255).astype('uint8')
                masks = (masks * 127.5 + 127.5).clip(0, 255).astype('uint8')
                total = 2
                if args.pred:
                    pred_masks = pred_masks.cpu().detach().numpy()
                    pred_masks = (pred_masks * 127.5 + 127.5).clip(0, 255).astype('uint8')
                    total = 3
                fig = plt.figure(figsize=(10, 5))
                # plot one reconstructed and one original image
                plt.subplot(1, total, 1)
                plt.imshow(recon[0].transpose(1, 2, 0))
                plt.title("Reconstructed Mask")
                plt.axis('off')
                plt.subplot(1, total, 2)
                plt.imshow(masks[0].transpose(1, 2, 0))
                plt.title("Original Mask")
                plt.axis('off')
                if args.pred:
                    plt.subplot(1, total, 3)
                    plt.imshow(pred_masks[0].transpose(1, 2, 0))
                    plt.title("Predicted Mask")
                    plt.axis('off')

                wandb.log({"reconstructed_mask": fig})
                plt.close(fig)
                #
                # Save model checkpoint
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                torch.save(vae.decoder.state_dict(), os.path.join(models_dir, f"vae_decoder_step_{cnt}_{args.dataset}_{args.size}.pt"))

            cnt += 1

            if cnt-1 == args.n_steps:
                break
        if cnt-1 == args.n_steps:
            break

        avg_loss = epoch_loss / len(dataloader.dataset)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")