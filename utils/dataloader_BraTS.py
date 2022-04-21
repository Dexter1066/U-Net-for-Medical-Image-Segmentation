# todo: add relevant libs
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
from PIL import Image, ImageOps
from utils.DataAugmentation_BraTS import get_train_transform, get_val_transform


SEGMENT_CLASS = {
   0: 'NOT TUMOR',
   1: 'NECROTIC/CORE',
   2: 'EDEMA',
   4: 'ENHANCING'
}

MODALITIES = ['t1', 't1ce', 't2', 'flair']

VOLUME_SLICES = 155
VOLUME_START_AT = 22
IMG_SIZE = 128

TRAIN_PATH = "../data/train/"
VALIDATION_PATH = "../data/val/"


def process_f32(img_path):
    """Set all voxels outside the brain mask to 0"""
    name = os.path.basename(img_path)
    images = np.stack([
        np.array(nib_load(os.path.join(img_path, name) + '_' + i + 'nii.gz'),
                 dtype='float32', order='C') for i in MODALITIES], -1
    )
    mask = images.sum(-1) > 0

    for k in range(len(MODALITIES)):
        x = images[..., k]
        y = x[mask]

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & x < lower] = lower
        x[mask & x > upper] = upper

        y = x[mask]
        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    return images


def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


class DataGenerator(Dataset):
    def __init__(self, data_dir, split='train', case_name=[], transform=None):
        self.list_IDs = list_IDs
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.list_IDs[index]

