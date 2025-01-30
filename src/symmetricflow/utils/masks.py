import torch
import numpy as np

def mask_to_class(masks, dataset):
    '''
    Converts a mask to a class label
    Args:
        masks: torch.Tensor
        dataset: str
    Returns:
        torch.Tensor: class labels
    '''

    # convert masks to numpy arrays, and then normalize them from 0,1 to 255 as integers
    masks = masks.numpy()
    masks = (masks * 255).astype(np.uint8)

    if dataset == 'celeba':
        color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    # convert masks to class labels by measuring the distance to the color in the color_list
    class_distances = np.zeros((masks.shape[0], masks.shape[1], len(color_list)))

    for i, color in enumerate(color_list):
        class_distances[:,:,i] = np.linalg.norm(masks - color, axis=-1)

    class_labels = np.argmin(class_distances, axis=-1)

    return class_labels