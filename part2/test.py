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
def load_model(pth):
    model = Network2(3, 1)
    model.load_state_dict(torch.load(pth, map_location=device))
    return model


# model_paths = ['./parameters75.pth', './parameters460.pth', './parameters291.pth', './parameters460.pth']
# model_paths = ['./CE.pth', './dice.pth', './diceCE.pth']
model_paths = ['./segmentation1.pth']#, './segmentation2.pth']
models = [load_model(pth).to(device) for pth in model_paths]

cap = cv2.VideoCapture('../test_video.mp4')
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)
result = cv2.VideoWriter('../output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (2 * w, len(models) * h))
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
    rgb_in = F.resize(rgb, [256, 256], interpolation=F.InterpolationMode.NEAREST_EXACT).unsqueeze(0)
    print(models[0](rgb_in).detach().shape)
    preds = [
        F.resize(
            model(rgb_in).detach().squeeze(0),
            [h, w], 
            interpolation=F.InterpolationMode.NEAREST_EXACT
        ) for model in models
    ]
    
    final_imgs = [utils.combine_test_images(pred, rgb) for pred in preds]
    final_img = np.concatenate(final_imgs, axis=0)
    # final_img = final_imgs[0]
    

    # cv2.imshow('frame', new_pred)
    # cv2.imshow('original', frame)
    cv2.imshow('combined', final_img)
    cv2.imwrite('../test.png', final_img)
    result.write(final_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()