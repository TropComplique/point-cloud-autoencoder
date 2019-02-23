import os
import numpy as np
import math
import torch
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
