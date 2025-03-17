import torch
import numpy as np
import matplotlib.pyplot as plt

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

def mask_to_class(masks, dataset):
    '''
    Converts a mask to a class label using PyTorch.
    Args:
        masks: torch.Tensor of shape (B, C, H, W) with values in [-1,1]
        dataset: str, either 'celeba' or 'coco'
    Returns:
        torch.Tensor: class labels of shape (B, H, W)
    '''
    # Normalize from [-1,1] to [0,1]
    masks = masks * 0.5 + 0.5
    masks = masks.clamp(0, 1)  # Ensure values are in [0,1]

    # Convert to [0, 255] and cast to float
    masks = (masks * 255).to(torch.float32)

    # Define color lists
    if dataset == 'celeba':
        color_list = torch.tensor([
            [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], 
            [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], 
            [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], 
            [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
        ], dtype=torch.float32, device=masks.device)  # Convert to float32

    elif dataset == 'coco':
        color_list = build_palette(6, 50)  # Assuming build_palette returns a list of RGB colors
        color_list = color_list[:171] + color_list[-1:]  # Trim specific indices
        color_list = torch.tensor(color_list, dtype=torch.float32, device=masks.device)  # Convert to tenso

    # Reshape color list for broadcasting: (Classes, 3, 1, 1)
    color_list = color_list.view(-1, 3, 1, 1)

    # Compute Euclidean distance for each pixel and each class
    distances = torch.norm(masks.unsqueeze(1) - color_list, dim=2)  # (B, Classes, H, W)

    # Assign each pixel to the closest class
    class_labels = torch.argmin(distances, dim=1)  # (B, H, W)

    return class_labels  # Shape: (B, H, W), dtype=torch.long


#def mask_to_class(masks, dataset):
    '''
    Converts a mask to a class label
    Args:
        masks: torch.Tensor
        dataset: str
    Returns:
        torch.Tensor: class labels
    '''
'''
    # convert masks to numpy arrays, and then normalize them from 0,1 to 255 as integers
    masks = masks.numpy()
    #normalize them back to 0,1
    masks = masks*0.5 + 0.5
    masks = np.clip(masks, 0, 1)
    #convert to 0,255
    masks = (masks * 255).astype(np.uint8)

    if dataset == 'celeba':
        color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    elif dataset == 'coco':
        color_list = build_palette(6,50)
        # all elements between indices 170 and the second to last element should be removed, but you must include the last element
        color_list = color_list[:170] + color_list[-1:]

    # masks have chape B x C x H x W, we want to get results of shape B x Classes x H x W
    color_array = np.array(color_list)  # Shape: (Classes, 3)

    # create a new tensor to store the distances to each color (class)
    distances = torch.zeros(masks.shape[0], len(color_list), masks.shape[2], masks.shape[3])

    # iterate over the color list and determine the distance to each color for every pixel
    # Initialize distance tensor
    distances = np.zeros((masks.shape[0], len(color_list), masks.shape[2], masks.shape[3]))  # (B, Classes, H, W)

    # Compute distances from each pixel to each color
    for i, color in enumerate(color_array):
        color_diff = masks - color.reshape(1, 3, 1, 1)  # Broadcast color across image dimensions
        distances[:, i] = np.linalg.norm(color_diff, axis=1)  # Euclidean distance over RGB channels

    # Assign each pixel to the closest class
    class_labels = np.argmin(distances, axis=1)  # Shape: (B, H, W)

    return torch.tensor(class_labels, dtype=torch.long)

'''  