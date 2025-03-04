import os
import random

import numpy
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.io import loadmat
from skimage import transform as sk_transform
from pywt import dwt, idwt

from utils import normalize, Mask, PCA, band_selection


class make_dataset(Dataset):
    def __init__(self, root, with_label, channels, transform, size=None):
        super(make_dataset, self).__init__()
        self.root = root
        self.with_label = with_label
        self.filelist = os.listdir(root)
        self.channels = channels
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        data, label = _get_data(self.root + self.filelist[item], self.with_label, self.channels, transform=self.transform, size=self.size)

        return data, label, self.filelist[item]


def get_dataloader(root='/home/worker1/DATASETS/HAD100/mat/without_anomaly/', batch_size=1, with_label=False, channels=None, shuffle=True, transform=True, size=None):
    assert os.path.isdir(root), 'input root is not dir'
    dataset = make_dataset(root, with_label, channels, transform, size)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)

    return dataloader


def _get_data(path, with_label, channels, data_key='data', label_key='map', transform=True, size=None):
    mat_data = loadmat(path)

    if with_label:
        if size is not None:
            label = np.expand_dims(mat_data[label_key], axis=-1)[0:size, 0:size, :]
            if np.max(label) == 0:
                label = np.expand_dims(mat_data[label_key], axis=-1)[-size:, -size:, :]
        else:
            label = np.expand_dims(mat_data[label_key], axis=-1)
    else:
        generator = Mask()
        if size is not None:
            label = (1 - np.expand_dims(generator(1)[0], axis=-1))[0:size, 0:size, :]
        else:
            label = (1 - np.expand_dims(generator(1)[0], axis=-1))

    if size is not None:
        if channels is None:
            data = normalize(mat_data[data_key][0:size, 0:size, :])
        else:
            data = normalize(mat_data[data_key][0:size, 0:size, 0:channels])
    else:
        if channels is None:
            data = normalize(mat_data[data_key])
        else:
            data = (mat_data[data_key][:, :, 0:channels])
            # data = normalize(PCA(mat_data[data_key].astype(np.float32), channels))
            # data = normalize(band_selection(mat_data[data_key].astype(np.float32), channels))

    if transform:
        # rotate
        data = sk_transform.rotate(data, random.choice([0, 90, 180, 270]))
        # flip
        if random.random() > 0.5:
            data = data[:, ::-1, ...].copy()
        if random.random() > 0.5:
            data = data[::-1, :, ...].copy()

    data = np.transpose(data, (2, 0, 1)).astype(np.float32)
    label = np.transpose(label, (2, 0, 1)).astype(np.float32)

    return data, label

