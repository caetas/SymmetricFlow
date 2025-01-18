from data.Dataloaders import celeb_hq_masked_dataloader
from utils.util import parse_args_SymmetricFlowMatching
from models.SymmFM import SymmFM

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatching()
    image_shape, channels, dataloader = celeb_hq_masked_dataloader(1, 0, 'train', args.size)
    _, _, dataloader_val = celeb_hq_masked_dataloader(16, 0, 'validation', args.size)

    model = SymmFM(args, image_shape, channels)
    model.train_model(dataloader, dataloader_val)