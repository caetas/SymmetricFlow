from data.Dataloaders import celeb_hq_masked_dataloader, cityscapes_dataloader, cocostuff_dataloader, ade20k_dataloader
from utils.util import parse_args_SymmetricFlowMatching
from models.SymmFM import SymmFM
import torch

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatching()

    if args.train:
        if args.dataset == 'celeba':
            image_shape, channels, dataloader = celeb_hq_masked_dataloader(args.batch_size, args.num_workers, 'train', args.size)
            _, _, dataloader_val = celeb_hq_masked_dataloader(16, args.num_workers, 'validation', args.size)
        elif args.dataset == 'cityscapes':
            image_shape, channels, dataloader = cityscapes_dataloader(args.batch_size, args.num_workers, 'train', args.size)
            _, _, dataloader_val = cityscapes_dataloader(16, args.num_workers, 'validation', args.size)
        elif args.dataset == 'ade20k':
            image_shape, channels, dataloader = ade20k_dataloader(args.batch_size, args.num_workers, 'train', args.size)
            _, _, dataloader_val = ade20k_dataloader(16, args.num_workers, 'validation', args.size)
        else:
            image_shape, channels, dataloader = cocostuff_dataloader(args.batch_size, args.num_workers, 'train', args.size)
            _, _, dataloader_val = cocostuff_dataloader(16, args.num_workers, 'val', args.size)

        model = SymmFM(args, image_shape, channels)
        model.train_model(dataloader, dataloader_val)

    else:
        if args.dataset == 'celeba':
            image_shape, channels, dataloader = celeb_hq_masked_dataloader(16, args.num_workers, 'validation', args.size)
        elif args.dataset == 'cityscapes':
            image_shape, channels, dataloader = cityscapes_dataloader(16, args.num_workers, 'validation', args.size)
        elif args.dataset == 'ade20k':
            image_shape, channels, dataloader = ade20k_dataloader(16, args.num_workers, 'validation', args.size)
        else:
            image_shape, channels, dataloader = cocostuff_dataloader(16, args.num_workers, 'val', args.size)

        model = SymmFM(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)
        # get a batch from loader
        x, mask = next(iter(dataloader))
        x = x.to(model.device)
        mask = model.dequantize_mask(mask)
        mask = mask.to(model.device)
        if model.vae is not None:
            with torch.no_grad():
                # if x has one channel, make it 3 channels
                if x.shape[1] == 1:
                    x = torch.cat((x, x, x), dim=1)
                    mask = torch.cat((mask, mask, mask), dim=1)
                #x = self.vae.module.encode(x).latent_dist.sample().mul_(0.18215)
                x = model.encode(x).latent_dist.sample().mul_(0.18215)
                #mask = self.vae.module.encode(mask).latent_dist.mode().mul_(0.18215)
                mask = model.encode(mask).latent_dist.mode().mul_(0.18215)
        
        model.sample(16, mask, train=False)
        model.segment(16, x, train=False)