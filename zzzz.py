'''
@Author: rayenwang
@Date: 2019-07-18 10:38:45
@LastEditTime: 2019-07-19 19:01:35
@Description: 
'''
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F

# image = cv2.imread('../dataset/MOT17/train/MOT17-02-DPM/img1/000001.jpg')
# image = torch.from_numpy(image).float().unsqueeze(0).permute((0, 3, 1, 2))
# h, w = image.shape[2:]
# bbox = np.asarray([912, 484, 97, 109])
# x, y, w, h = bbox / np.asarray([w, h, w, h])
# theta = np.asarray([
#     [w, 0, -1 + 2 * x + w],
#     [0, h, -1 + 2 * y + h]
# ])


# theta = torch.from_numpy(theta).float().unsqueeze(0)
# grid = F.affine_grid(theta, [1, 3, 512, 512])
# out = F.grid_sample(image, grid)
# out = out.permute((0, 2, 3, 1))[0].numpy().astype(np.uint8)
# print(out.shape)
# cv2.namedWindow('win', cv2.WINDOW_NORMAL)
# cv2.imshow('win', out)
# cv2.waitKey()
from config import TrainConfig as Config
torch.device('cuda:0')
#torch.load(Config.backbone, map_location='gpu')
