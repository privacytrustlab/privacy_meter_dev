import pdb

import numpy as np
import torchvision.transforms as T
import pickle

# Define a mapping from string names to torchvision augmentation functions
augmentation_mapping = {
    "horizontal_flip": T.RandomHorizontalFlip(p=1.0),
    "vertical_flip": T.RandomVerticalFlip(p=1.0),
    "rotate": T.RandomRotation(degrees=30),
}


def sample_geometric(range_center, transformations_list, sample_size):
    """
    Sample points in the geometric range of the range_center.

    Args:
        range_center (np.ndarray): The center of the range.
        transformations_list (list): A list of strings representing the transformations to apply.
        sample_size (int): The number of samples to generate.

    Returns:
        list[torch.tensors]: The samples in the geometric range.
    """
    # Initialize the samples list
    if len(transformations_list) == sample_size - 1:
        samples = [range_center]  # Include the range_center as the first sample
    else:
        samples = []

    # Apply each transformation to the range_center
    for transformation in transformations_list:
        if transformation in augmentation_mapping:
            samples.append(augmentation_mapping[transformation](range_center))
        else:
            raise ValueError(f"Invalid transformation: {transformation}")

    return samples
