# todo: add relevant libs
import os
import numpy as np
import torch.utils.data as data
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
    def __init__(self, data_dir, split='train', case_names=[], transform=None):
        super(DataGenerator, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.case_names = case_names
        self.transform = transform

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        name = self.case_names[index]
        base_dir = os.path.join(self.data_dir, name)

        flair = np.array(nib_load(base_dir + '_flair.nii.gz'), dtype='float32')
        t1 = np.array(nib_load(base_dir + '_t1.nii.gz'), dtype='float32')
        t1ce = np.array(nib_load(base_dir + '_t1ce.nii.gz'), dtype='float32')
        t2 = np.array(nib_load(base_dir + '_t2.nii.gz'), dtype='float32')
        mask = np.array(nib_load(base_dir + '_seg.nii.gz'), dtype='uint8')

        if self.split == 'train':
            item = self.transform({'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'label': mask})[0]
        elif self.split == 'val':
            item = self.transform({'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'label': mask})
        else:
            raise NotImplementedError

        return item['image'], item['label'], index, name


def train_loader(data_path, case_names, train_batch_size, num_workers, patch_size=128, pos_ratio=1.0, neg_ratio=1):
    train_transform = get_train_transform(patch_size, pos_ratio, neg_ratio)
    train_data = DataGenerator(data_path, 'train', case_names, train_transform)

    return data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=False,
                           num_workers=num_workers, pin_memory=True)


def val_loader(data_path, case_names, val_batch_size, num_workers):
    val_transform = get_train_transform()
    val_data = DataGenerator(data_path, 'val', case_names, val_transform)

    return data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True, drop_last=False,
                           num_workers=num_workers, pin_memory=True)

