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


def cityscapes_to_color_mask(mask):
    id_to_color = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    -1: (0, 0, 142),
}

    mask = np.array(mask)
    mask = mask.squeeze()

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for id, color in id_to_color.items():
        colored_mask[np.all(mask == id, axis=-1)] = color

    return Image.fromarray(colored_mask)

class CityscapesDataset(Dataset):
    def __init__(self, transform, transform_mask, mode='train'):
        '''
        Initializes the CityscapesDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.dataset = load_dataset('Chris1/cityscapes', split=mode)
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
        image = example['image']
        mask = example['semantic_segmentation']
        # crop the central 1024x1024 region, remember its a PIL image of shape 2048x1024
        image = image.crop((512, 0, 1536, 1024))
        mask = mask.crop((512, 0, 1536, 1024))
        mask = cityscapes_to_color_mask(mask)
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