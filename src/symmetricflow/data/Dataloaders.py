from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch
import numpy as np
from PIL import Image
from io import BytesIO

class CelebHQMaskedDataset(Dataset):
    def __init__(self, transform_fn, mode='train'):
        '''
        Initializes the CelebHQMaskedDataset
        Args:
            transform_fn: function
            mode: str
        '''

        self.dataset = load_dataset('eurecom-ds/celeba_hq_mask', split=mode)
        self.transform_fn = transform_fn
        self.dataset.set_transform(self.transform_fn)
        #self.dataset = self.dataset[mode]

    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        # get self.patches random patches from the image
        image = self.dataset[idx]['pixel_values']
        mask = self.dataset[idx]['mask_values']
        # dequantize by adding +- up to a range of 1/255
        image = image + (torch.rand_like(image) - 0.5) / 127.5
        mask = mask
        return image, mask
    
def celeb_hq_masked_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the CelebHQMaskedDataset
    Args:
        batch_size: int
        num_workers: int
        mode: str
        input_shape: int
    Returns:
        DataLoader
    '''
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((input_shape,input_shape), interpolation=transforms.InterpolationMode.NEAREST) if input_shape is not None else transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    def transform_fn(examples):
        examples['pixel_values'] = [transform(image) for image in examples['image']]
        examples['mask_values'] = [transform_mask(mask) for mask in examples['mask']]
        return examples

    dataset = CelebHQMaskedDataset(transform_fn=transform_fn, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True if mode != 'train' else False)

    return input_shape, 3, dataloader

class CityscapesDataset(Dataset):
    def __init__(self, transform, transform_mask, mode='train'):
        '''
        Initializes the CityscapesDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.dataset = load_dataset('huggan/cityscapes', split='train')
        self.transform = transform
        self.transform_mask = transform_mask
        #self.dataset = self.transform_fn(self.dataset)  # Apply transformation to the entire dataset
        self.mode = mode

    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        example = self.dataset[idx]
        image = example['imageA']['bytes']
        mask = example['imageB']['bytes']
        image = Image.open(BytesIO(image))
        mask = Image.open(BytesIO(mask))
        image = self.transform(image)
        mask = self.transform_mask(mask)
        if self.mode == 'train':
            # flip the image and mask horizontally with a 50% chance
            if np.random.rand() > 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
        return image, mask


def cityscapes_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the CityscapesDataset
    Args:
        batch_size: int
        num_workers: int
        mode: str
        input_shape: int
    Returns:
        DataLoader
    '''
    transform = transforms.Compose([
        transforms.Resize((input_shape, input_shape)) if input_shape is not None else transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((input_shape, input_shape), interpolation=transforms.InterpolationMode.NEAREST) if input_shape is not None else transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



    dataset = CityscapesDataset(transform=transform, transform_mask=transform_mask, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=(mode != 'train'))

    return input_shape, 3, dataloader