import cv2
import torch
import os
from torch.utils.data import Dataset
import glob
import random

from torch.utils.data.dataset import T_co


class DataLoader(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(data_path, 'images/*.tif'))
        # print(data_path)
        # print(self.img_path)

    def augment(self, image, flipcode):
        return cv2.flip(image, flipcode)
    
    def __getitem__(self, index):
        image_path = self.img_path[index]
        # set the label path according to the image path
        label_path = image_path.replace('images', 'label')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # convert the image to single channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], image.shape[1])

        if label.max() > 1:
            label = label / 255

        # randomly data augmentation
        flipCode = random.choice([1, 0, -1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    isbi_dataset = DataLoader("../data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)