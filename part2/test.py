# IMPORTS----------------------------------------------------------------------------
# STANDARD
import sys
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Subset

import wandb

# CUSTOM
from network import *
from network2 import *
from utils import *
from dataloader import *
import pdb
import utils
from  torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor, Resize
import torch.nn.functional as Fnn
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

# Load the parameters
from loadParam import *

# DATASET ---------------------------------------------------------------------------
datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device', device)

# Network and optimzer --------------------------------------------------------------
model = Network2(3, 1)

model = model.to(device)
model.load_state_dict(torch.load("./parameters68.pth", map_location=device))

cap = cv2.VideoCapture('../test_video.mp4')
result = cv2.VideoWriter('../output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (512, 256))
final_imgs = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print('finished, saving...')

        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb_frame)

    # convert them to tensors
    rgb = transforms.ToTensor()(im_pil).to(device) # (3, H, W)
    rgb = F.resize(rgb, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT).unsqueeze(0)
    pred = model(rgb).detach().squeeze()
    
    print(torch.max(pred))
    new_pred = Fnn.sigmoid(pred)
    print(torch.max(new_pred), torch.min(new_pred))
    new_pred[new_pred > 0.5] = 1
    new_pred[new_pred <= 0.5] = 0
    
    new_pred = new_pred.cpu().numpy().astype(np.uint8) #* 255
    new_pred = cv2.cvtColor(new_pred, cv2.COLOR_GRAY2RGB)
    print(np.max(new_pred), np.min(new_pred))

    rgb = rgb.detach().cpu().numpy().squeeze()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    print(rgb.shape, new_pred.shape)

    final_img = np.concatenate((new_pred, rgb), axis=1)
    final_img = (np.clip(final_img, 0, 1)*255).astype(np.uint8)

    # cv2.imshow('frame', new_pred)
    # cv2.imshow('original', frame)
    cv2.imshow('combined', final_img)
    cv2.imwrite('../test.png', final_img)
    result.write(final_img)
    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()