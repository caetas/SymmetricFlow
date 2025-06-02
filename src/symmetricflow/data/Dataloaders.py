from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from tqdm import tqdm
from torchvision import datasets
from config import data_raw_dir
import zipfile
import os
from glob import glob

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

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
    def __init__(self, transform, transform_mask, mode='train', input_shape=256):
        '''
        Initializes the CityscapesDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.dataset = load_dataset('Chris1/cityscapes', split=mode)
        self.transform = transform
        self.transform_mask = transform_mask
        self.input_shape = input_shape
        self.mode = mode
        if self.mode == 'train':
            # resize the images to the input_shape with bilinear interpolation
            self.images = [data['image'].resize((self.input_shape*2, self.input_shape)) for data in tqdm(self.dataset, desc='Loading Cityscapes Dataset Images', leave=False)]
            # resize the masks to the input shape with nearest neighbor interpolation
            self.masks = [cityscapes_to_color_mask(data['semantic_segmentation'].resize((self.input_shape*2, self.input_shape), resample=Image.NEAREST)) for data in tqdm(self.dataset, desc='Loading Cityscapes Dataset Masks', leave=False)]
        else:
            # resize the images to the input_shape with bilinear interpolation and center crop
            self.images = [data['image'].resize((self.input_shape*2, self.input_shape)).crop((self.input_shape//2, 0, self.input_shape + self.input_shape//2, self.input_shape)) for data in tqdm(self.dataset, desc='Loading Cityscapes Dataset Images', leave=False)]
            self.masks = [cityscapes_to_color_mask(data['semantic_segmentation'].resize((self.input_shape*2, self.input_shape), resample=Image.NEAREST).crop((self.input_shape//2, 0, self.input_shape + self.input_shape//2, self.input_shape))) for data in tqdm(self.dataset, desc='Loading Cityscapes Dataset Masks', leave=False)]

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
        image = self.images[idx]
        mask = self.masks[idx]
        if self.mode == 'train':
            # random horizontal crop of the image and mask
            w_init = np.random.randint(0, self.input_shape)
            image = image.crop((w_init, 0, w_init+self.input_shape, self.input_shape))
            mask = mask.crop((w_init, 0, w_init+self.input_shape, self.input_shape))
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



    dataset = CityscapesDataset(transform=transform, transform_mask=transform_mask, mode=mode, input_shape=input_shape if input_shape is not None else 256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    return input_shape, 3, dataloader

def build_palette(k=6,s=None):
    if s==None:
        s = 250 // (k-1)
    else:
        assert s*(k-1)<255
    palette = []
    for m0 in range(k):
        for m1 in range(k):
            for m2 in range(k):
                palette.extend([s*m0,s*m1,s*m2])

    palette = [palette[i:i+3] for i in range(0, len(palette), 3)]

    return palette

class ADE20KDataset(Dataset):
    def __init__(self, transform_fn, mask_transform_fn, mode='train', size=256):
        '''
        Initializes the ADE20KDataset
        Args:
            transform_fn: function
            mode: str
        '''
        if mode == 'train':
            mode = 'training'
        elif mode == 'val':
            mode = 'validation'
        self.file_path = os.path.join(data_raw_dir, 'ADEChallengeData2016', f'images' ,f'{mode}')
        self.masks_file_path = os.path.join(data_raw_dir, 'ADEChallengeData2016', f'annotations' ,f'{mode}')
        self.transform_fn = transform_fn
        self.mode = mode
        self.image_ids = []
        self.mask_transform_fn = mask_transform_fn
        self.masks_ids = []
        self.image_list = os.listdir(self.file_path)
        self.masks = os.listdir(self.masks_file_path)
        self.images = [os.path.join(self.file_path, image) for image in self.image_list]
        #masks have the same name as the images but with a different extension (.png)
        self.masks = [os.path.join(self.masks_file_path, image.replace('.jpg', '.png')) for image in self.image_list]
        self.size = size
        self.palette = build_palette(6, 50)      
        print(f'Found {len(self.images)} images and {len(self.masks)} masks in the {mode} dataset')


    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.images)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        # if mask is a list, take the first one
        if isinstance(mask, list):
            mask = mask[0]
        # crop both to the smallest dimension, check if it is height or width, should be a center crop
        if image.size[0] < image.size[1]:
            start = np.random.randint(0, image.size[1] - image.size[0])
            image = image.crop((0, start, image.size[0], start + image.size[0]))
            mask = mask.crop((0, start, mask.size[0], start + mask.size[0]))
        elif image.size[0] > image.size[1]:
            start = np.random.randint(0, image.size[0] - image.size[1])
            image = image.crop((start, 0, start + image.size[1], image.size[1]))
            mask = mask.crop((start, 0, start + mask.size[1], mask.size[1]))
        
        # resize the image and mask to the input_shape
        image = image.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)
        mask = self.mask_to_color(mask)
        image = self.transform_fn(image)
        mask = self.mask_transform_fn(mask)
        return image, mask
    
    def mask_to_color(self, mask):
        mask = np.array(mask)
        mask = mask.squeeze()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        #change this to get the original mask colors
        for unique in np.unique(mask):
            colored_mask[mask == unique] = self.palette[self.mapping(unique)]
        return Image.fromarray(colored_mask)
    
    def mapping(self, index):
        if index == 0:
            return len(self.palette) - 1  # background
        else:
            return index - 1
    
def ade20k_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the ADE20KDataset
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

    input_shape = input_shape if input_shape is not None else 256

    dataset = ADE20KDataset(transform_fn=transform, mask_transform_fn=transform_mask, mode=mode, size=input_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    return input_shape, 3, dataloader

class CocoStuffDataset(Dataset):
    
    def __init__(self, transform_fn, mask_transform_fn, mode='train', size=256):
        '''
        Initializes the CocoStuffDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.file_path = os.path.join(data_raw_dir, 'cocostuff', f'{mode}2017')
        self.masks_file_path = os.path.join(data_raw_dir, 'cocostuff', 'stuffthingmaps_trainval2017', f'{mode}2017')
        self.transform_fn = transform_fn
        self.mode = mode
        self.image_ids = []
        self.mask_transform_fn = mask_transform_fn
        self.masks_ids = []
        self.image_list = os.listdir(self.file_path)
        self.masks = os.listdir(self.masks_file_path)
        self.images = [os.path.join(self.file_path, image) for image in self.image_list]
        #masks have the same name as the images but with a different extension (.png)
        self.masks = [os.path.join(self.masks_file_path, image.replace('.jpg', '.png')) for image in self.image_list]
        self.palette = build_palette(6, 50)
        self.mappings = self.build_mapping()
        self.size = size

        print(f'Found {len(self.images)} images and {len(self.masks)} masks in the {mode} dataset')

    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.images)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        '''
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_file:
            with zip_file.open(self.image_ids[idx]) as image_file:
                image = Image.open(BytesIO(image_file.read()))
        with zipfile.ZipFile(self.masks_zip_file_path, 'r') as masks_zip_file:
            with masks_zip_file.open(self.masks_ids[idx]) as mask_file:
                mask = Image.open(BytesIO(mask_file.read()))
        '''
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        # crop both to the smallest dimension, check if it is height or width, should be a center crop
        if image.size[0] < image.size[1]:
            start = np.random.randint(0, image.size[1] - image.size[0])
            image = image.crop((0, start, image.size[0], start + image.size[0]))
            mask = mask.crop((0, start, mask.size[0], start + image.size[0]))
        elif image.size[0] > image.size[1]:
            start = np.random.randint(0, image.size[0] - image.size[1])
            image = image.crop((start, 0, start + image.size[1], image.size[1]))
            mask = mask.crop((start, 0, start + image.size[1], image.size[1]))

        # resize the image and mask to the input_shape
        image = image.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)
        mask = self.mask_to_color(mask)
        image = self.transform_fn(image)
        mask = self.mask_transform_fn(mask)
        return image, mask
    
    def mask_to_color(self, mask):
        mask = np.array(mask)
        mask = mask.squeeze()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        #change this to get the original mask colors
        for unique in np.unique(mask):
            colored_mask[mask == unique] = self.palette[self.mappings[unique]]
        return Image.fromarray(colored_mask)

    def build_mapping(self):
        '''
        Builds the mapping from the original mask to the new mask
        Returns:
            dict: mapping
        '''
        mappings = {0 : 0 ,
            1 : 1 ,
            2 : 2 ,
            3 : 3 ,
            4 : 4 ,
            5 : 5 ,
            6 : 6 ,
            7 : 7 ,
            8 : 8 ,
            9 : 9 ,
            10 : 10 ,
            12 : 11 ,
            13 : 12 ,
            14 : 13 ,
            15 : 14 ,
            16 : 15 ,
            17 : 16 ,
            18 : 17 ,
            19 : 18 ,
            20 : 19 ,
            21 : 20 ,
            22 : 21 ,
            23 : 22 ,
            24 : 23 ,
            26 : 24 ,
            27 : 25 ,
            30 : 26 ,
            31 : 27 ,
            32 : 28 ,
            33 : 29 ,
            34 : 30 ,
            35 : 31 ,
            36 : 32 ,
            37 : 33 ,
            38 : 34 ,
            39 : 35 ,
            40 : 36 ,
            41 : 37 ,
            42 : 38 ,
            43 : 39 ,
            45 : 40 ,
            46 : 41 ,
            47 : 42 ,
            48 : 43 ,
            49 : 44 ,
            50 : 45 ,
            51 : 46 ,
            52 : 47 ,
            53 : 48 ,
            54 : 49 ,
            55 : 50 ,
            56 : 51 ,
            57 : 52 ,
            58 : 53 ,
            59 : 54 ,
            60 : 55 ,
            61 : 56 ,
            62 : 57 ,
            63 : 58 ,
            64 : 59 ,
            66 : 60 ,
            69 : 61 ,
            71 : 62 ,
            72 : 63 ,
            73 : 64 ,
            74 : 65 ,
            75 : 66 ,
            76 : 67 ,
            77 : 68 ,
            78 : 69 ,
            79 : 70 ,
            80 : 71 ,
            81 : 72 ,
            83 : 73 ,
            84 : 74 ,
            85 : 75 ,
            86 : 76 ,
            87 : 77 ,
            88 : 78 ,
            89 : 79 ,
            91 : 80 ,
            92 : 81 ,
            93 : 82 ,
            94 : 83 ,
            95 : 84 ,
            96 : 85 ,
            97 : 86 ,
            98 : 87 ,
            99 : 88 ,
            100 : 89 ,
            101 : 90 ,
            102 : 91 ,
            103 : 92 ,
            104 : 93 ,
            105 : 94 ,
            106 : 95 ,
            107 : 96 ,
            108 : 97 ,
            109 : 98 ,
            110 : 99 ,
            111 : 100 ,
            112 : 101 ,
            113 : 102 ,
            114 : 103 ,
            115 : 104 ,
            116 : 105 ,
            117 : 106 ,
            118 : 107 ,
            119 : 108 ,
            120 : 109 ,
            121 : 110 ,
            122 : 111 ,
            123 : 112 ,
            124 : 113 ,
            125 : 114 ,
            126 : 115 ,
            127 : 116 ,
            128 : 117 ,
            129 : 118 ,
            130 : 119 ,
            131 : 120 ,
            132 : 121 ,
            133 : 122 ,
            134 : 123 ,
            135 : 124 ,
            136 : 125 ,
            137 : 126 ,
            138 : 127 ,
            139 : 128 ,
            140 : 129 ,
            141 : 130 ,
            142 : 131 ,
            143 : 132 ,
            144 : 133 ,
            145 : 134 ,
            146 : 135 ,
            147 : 136 ,
            148 : 137 ,
            149 : 138 ,
            150 : 139 ,
            151 : 140 ,
            152 : 141 ,
            153 : 142 ,
            154 : 143 ,
            155 : 144 ,
            156 : 145 ,
            157 : 146 ,
            158 : 147 ,
            159 : 148 ,
            160 : 149 ,
            161 : 150 ,
            162 : 151 ,
            163 : 152 ,
            164 : 153 ,
            165 : 154 ,
            166 : 155 ,
            167 : 156 ,
            168 : 157 ,
            169 : 158 ,
            170 : 159 ,
            171 : 160 ,
            172 : 161 ,
            173 : 162 ,
            174 : 163 ,
            175 : 164 ,
            176 : 165 ,
            177 : 166 ,
            178 : 167 ,
            179 : 168 ,
            180 : 169 ,
            181 : 170 ,
            255 : len(self.palette) - 1}
        return mappings  


def cocostuff_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the CocoStuffDataset
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

    input_shape = input_shape if input_shape is not None else 256

    dataset = CocoStuffDataset(transform_fn=transform, mask_transform_fn=transform_mask, mode=mode, size=input_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=(mode != 'train'))

    return input_shape, 3, dataloader

class SurgeSAMDataset(Dataset):
    def __init__(self, transform_fn, mask_transform_fn, mode='train', size=256):
        '''
        Initializes the SurgeSAMDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.images = glob(os.path.join(data_raw_dir, "SurgeSAM_processed", "*", "*", "*", "images", "*.jpg"))
        self.images.sort()
        val_videos = ['s8k98gGeFf0.mp4', 'watch#v=xEps8nqblY0.mp4']
        if mode == 'train':
            self.images = [img for img in self.images if not any(video in img for video in val_videos)]
        elif mode == 'val':
            self.images = [img for img in self.images if any(video in img for video in val_videos)]
        
        self.masks = [img.replace("images", "machine_masks").replace(".jpg", ".png") for img in self.images]
        self.transform_fn = transform_fn
        self.mask_transform_fn = mask_transform_fn
        self.size = size
        self.mode = mode
        self.palette = build_palette(4, 75)

    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.images)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        # Find smallest dimension
        min_dim = min(image.size)

        # Randomly choose a starting point within the allowed range for both width and height
        left = np.random.randint(0, image.size[0] - min_dim + 1) if image.size[0] > min_dim else 0
        top = np.random.randint(0, image.size[1] - min_dim + 1) if image.size[1] > min_dim else 0

        right = left + min_dim
        bottom = top + min_dim

        # Apply the random crop
        image = image.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        # resize the image and mask to the input_shape
        image = image.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)
        image = self.transform_fn(image)
        mask = self.mask_to_color(mask)
        mask = self.mask_transform_fn(mask)
        if self.mode == 'train':
            # flip the image and mask horizontally with a 50% chance
            if np.random.rand() > 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
        return image, mask

    def mask_to_color(self, mask):
        mask = np.array(mask)
        mask = mask.squeeze()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for unique in np.unique(mask):
            if unique == 255:
                unique = len(self.palette) - 1
                colored_mask[mask == 255] = self.palette[unique]
            else:
                colored_mask[mask == unique] = self.palette[unique]
        return Image.fromarray(colored_mask)
    
def surge_sam_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the SurgeSAMDataset
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

    input_shape = input_shape if input_shape is not None else 256

    dataset = SurgeSAMDataset(transform_fn=transform, mask_transform_fn=transform_mask, mode=mode, size=input_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=(mode != 'train'))

    return input_shape, 3, dataloader
        


def mnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers = num_workers)
    if input_shape is not None:
        return input_shape, 1, training_loader
    else:
        return 32, 1, training_loader

def mnist_val_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    validation_data = datasets.MNIST(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    if input_shape is not None:
        return input_shape, 1, validation_loader
    else:
        return 32, 1, validation_loader
    
def cifar10_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    training_data = datasets.CIFAR10(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers = num_workers)
    if input_shape is not None:
        return input_shape, 3, training_loader
    else:
        return 32, 3, training_loader
    
def cifar10_val_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    validation_data = datasets.CIFAR10(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    if input_shape is not None:
        return input_shape, 3, validation_loader
    else:
        return 32, 3, validation_loader