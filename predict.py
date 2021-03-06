import glob
import numpy as np
import os
import cv2
from model.unet import UNet
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(1, 1)
    model.to(device=device)

    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    tests_path = glob.glob('data/test/*.tif')
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.tif'
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)

        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        pred = model(img_tensor)

        pred = np.array(pred.data.cpu()[0])[0]

        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        cv2.imwrite(save_res_path, pred)