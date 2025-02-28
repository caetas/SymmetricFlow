from data.Dataloaders import mnist_train_loader, mnist_val_loader, cifar10_train_loader, cifar10_val_loader
from utils.util import parse_args_SymmetricFlowMatchingClass
from models.SymmFMClass import SymmFMClass
import torch

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatchingClass()

    if args.train:
        
        if args.dataset == 'mnist':
            image_shape, channels, dataloader = mnist_train_loader(args.batch_size, normalize=True, num_workers=args.num_workers)
            _, _, dataloader_val = mnist_val_loader(16, normalize=True)
        else:
            image_shape, channels, dataloader = cifar10_train_loader(args.batch_size, normalize=True, num_workers=args.num_workers)
            _, _, dataloader_val = cifar10_val_loader(16, normalize=True)

        model = SymmFMClass(args, image_shape, channels)
        #model.sample(16, mask=mask, train=False)
        #model.segment(16, x, train=False)
        model.train_model(dataloader, dataloader_val)

    elif args.sample:
        
        if args.dataset == 'mnist':
            image_shape, channels, dataloader = mnist_val_loader(16, normalize=True)
        else:
            image_shape, channels, dataloader = cifar10_val_loader(16, normalize=True)

        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)

        # create 16 masks for the 16 samples, based on the classes
        labels = torch.arange(0, args.num_samples).to(model.device) % args.n_classes
        mask = model.dequantize_class(labels)
        mask = mask.to(model.device)
        model.sample(args.num_samples, mask=mask, train=False)

    elif args.classification:
        
        if args.dataset == 'mnist':
            image_shape, channels, dataloader = mnist_val_loader(16, normalize=True)
        else:
            image_shape, channels, dataloader = cifar10_val_loader(16, normalize=True)

        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)
        model.evaluate_segmentation(dataloader)

    else:
        
        if args.dataset == 'mnist':
            image_shape, channels, _ = mnist_val_loader(args.batch_size, normalize=True)
        else:
            image_shape, channels, _ = cifar10_val_loader(args.batch_size, normalize=True)

        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)
        model.fid_sample(args.batch_size)