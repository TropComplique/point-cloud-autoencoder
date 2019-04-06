import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset


class PointClouds(Dataset):

    def __init__(self, coco, image_folder, is_training=False, training_size=None):
        """
        Arguments:
            is_training: a boolean.
            training_size: an integer or None.
        """
        self.is_training = is_training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        augmentation(x)
        return batch


def load_ply(filename):
    """
    Arguments:
        filename: a string.
    Returns:
        a float numpy array with shape [num_points, 3].
    """
    ply_data = PlyData.read(filename)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    return points


def augmentation(x):
    """
    Arguments:
        x: a float numpy array with shape [b, n, 3].
    Returns:
        a float numpy array with shape [b, n, 3].
    """

    jitter = np.random.normal(0.0, 1e-3, size=x.shape)
    x += jitter.astype('float32')

    # batch size
    b = x.shape[0]

    # random rotation matrix
    m = ortho_group.rvs(3, size=b)  # shape [b, 3, 3]
    m = np.expand_dims(m, 1)  # shape [b, 1, 3, 3]

    x = np.expand_dims(x, 3)
    x = np.matmul(x, m)
    x = np.squeeze(x, 3)

    return x
