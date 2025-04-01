from data.Dataloaders import surge_sam_dataloader
from utils.util import parse_args_SymmetricFlowMatching
from models.SymmFM import SymmFM
import torch

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatching()
    args.dataset = 'surgesam'

    if args.train:
        image_shape, channels, dataloader = surge_sam_dataloader(args.batch_size, args.num_workers, 'train', args.size)
        _, _, dataloader_val = surge_sam_dataloader(16, args.num_workers, 'val', args.size)

        model = SymmFM(args, image_shape, channels)
        model.train_model(dataloader, dataloader_val)

    elif args.sample:
        image_shape, channels, dataloader = surge_sam_dataloader(args.num_samples, args.num_workers, 'val', args.size)
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
                x = model.encode(x).latent_dist.sample().mul_(0.18215)
                mask = model.encode(mask).latent_dist.mode().mul_(0.18215)
        
        model.sample(args.num_samples, mask, train=False)
        model.segment(args.num_samples, x, train=False)