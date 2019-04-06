import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from scipy.stats import ortho_group


class PointClouds(Dataset):

    def __init__(self, dataset_path, labels, is_training=False):
        """
        Arguments:
            is_training: a boolean.
        """

        paths = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                p = os.path.join(path, name)
                assert p.endswith('.ply')
                paths.append(p)
        
        def get_label(p):
            return p.split('/')[-2]
        
        paths = [p for p in paths if get_label(p) in labels]
        self.is_training = is_training
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        p = self.paths[i]
        x = load_ply(p)
        
        x -= x.mean(0)
        d = np.sqrt((x ** 2).sum(1))
        x /= d.max()

        if self.is_training:
            x = augmentation(x)
        
        x = torch.FloatTensor(x).permute(1, 0)
        return x


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
    return points.astype('float32')


from scipy.stats import ortho_group

def augmentation(x):
    """
    Arguments:
        x: a float numpy array with shape [b, n, 3].
    Returns:
        a float numpy array with shape [b, n, 3].
    """

    jitter = np.random.normal(0.0, 1e-2, size=x.shape)
    x += jitter.astype('float32')

    # batch size
    b = x.shape[0]

    # random rotation matrix
    m = ortho_group.rvs(3)  # shape [b, 3, 3]
    m = np.expand_dims(m, 0)  # shape [b, 1, 3, 3]
    m = m.astype('float32')

    x = np.expand_dims(x, 1)
    x = np.matmul(x, m)
    x = np.squeeze(x, 1)

    return x
