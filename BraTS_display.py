import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import nilearn as nl
import nibabel as nib
import shutil
import nilearn.plotting as nlplt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize

TRAIN_PATH = "./data/MICCAI_BraTS2020_TrainingData/"

test_image_flair = nib.load(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1 = nib.load(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce = nib.load(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2 = nib.load(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask = nib.load(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


def show_whole_data():
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    slice_w = 25
    ax1.imshow(test_image_flair[:, :, test_image_flair.shape[0] // 2 - slice_w], cmap='gray')
    ax1.set_title('Image flair')
    ax2.imshow(test_image_t1[:, :, test_image_t1.shape[0] // 2 - slice_w], cmap='gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:, :, test_image_t1ce.shape[0] // 2 - slice_w], cmap='gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:, :, test_image_t2.shape[0] // 2 - slice_w], cmap='gray')
    ax4.set_title('Image t2')
    ax5.imshow(test_mask[:, :, test_mask.shape[0] // 2 - slice_w])
    ax5.set_title('Mask')
    plt.show()


def show_segment():
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(rotate(montage(test_image_t1[50:-50, :, :]), 90, resize=True), cmap='gray')
    plt.show()


def show_different_effect_segment():
    niimg = nl.image.load_img(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
    nimask = nl.image.load_img(TRAIN_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

    fig, axes = plt.subplots(nrows=4, figsize=(30, 40))

    nlplt.plot_anat(niimg,
                    title='BraTS20_Training_001_flair.nii plot_anat',
                    axes=axes[0])

    nlplt.plot_epi(niimg,
                   title='BraTS20_Training_001_flair.nii plot_epi',
                   axes=axes[1])

    nlplt.plot_img(niimg,
                   title='BraTS20_Training_001_flair.nii plot_img',
                   axes=axes[2])

    nlplt.plot_roi(nimask,
                   title='BraTS20_Training_001_flair.nii with mask plot_roi',
                   bg_img=niimg,
                   axes=axes[3], cmap='Paired')

    plt.show()


if __name__ == '__main__':
    show_whole_data()
    show_segment()
    show_different_effect_segment()
