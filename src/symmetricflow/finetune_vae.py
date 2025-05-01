from diffusers import AutoencoderKL
import torch
from data.Dataloaders import celeb_hq_masked_dataloader, cocostuff_dataloader
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import wandb
from config import models_dir
import os

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=2, help='batch size')
    argparser.add_argument('--dataset', type=str, default='celeba', help='dataset name', choices=['celeba', 'coco'])
    argparser.add_argument('--size', type=int, default=256, help='image size')
    argparser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    argparser.add_argument('--n_steps', type=int, default=2000, help='number of steps for training')
    return argparser.parse_args()

@torch.no_grad()
def validation_loss(vae, dataloader_val, criterion, device):
    val_loss = 0.0
    vae.decoder.eval()
    with torch.no_grad():
        for _, masks in tqdm(dataloader_val, desc="Validation Batches", leave=False):
            masks = masks.to(device)
            # perturb input with uniform noise
            masks = masks + 10*(torch.randn_like(masks) * 0.5)/127.5

            # Obtain latents from frozen encoder
            latents = vae.encode(masks).latent_dist.sample()

            # Decode to reconstructed masks
            recon = vae.decode(latents).sample

            # Compute loss against ground-truth masks
            loss = criterion(recon, masks)

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
        for _,masks in tqdm(dataloader, desc="Training Batches", leave=False):
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
                fig = plt.figure(figsize=(10, 5))
                # plot one reconstructed and one original image
                plt.subplot(1, 2, 1)
                plt.imshow(recon[0].transpose(1, 2, 0))
                plt.title("Reconstructed Mask")
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(masks[0].transpose(1, 2, 0))
                plt.title("Original Mask")
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