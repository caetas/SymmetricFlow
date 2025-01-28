from data.Dataloaders import celeb_hq_masked_dataloader, cityscapes_dataloader
from utils.util import parse_args_SymmetricFlowMatching
from models.SymmFM import SymmFM
import torch

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatching()
    if args.dataset == 'celeba':
        image_shape, channels, dataloader = celeb_hq_masked_dataloader(args.batch_size, args.num_workers, 'train', args.size)
        _, _, dataloader_val = celeb_hq_masked_dataloader(16, args.num_workers, 'validation', args.size)
    else:
        image_shape, channels, dataloader = cityscapes_dataloader(args.batch_size, args.num_workers, 'train', args.size)
        _, _, dataloader_val = cityscapes_dataloader(16, args.num_workers, 'validation', args.size)

    model = SymmFM(args, image_shape, channels)
    model.load_checkpoint(args.checkpoint)
    # one batch of dataloaderval
    x,mask= next(iter(dataloader))
    x = x.to(model.device)
    mask = mask.to(model.device)
    #model.sample(16, mask=mask, train=False)
    #model.segment(16, x, train=False)
    model.train_model(dataloader, dataloader_val)