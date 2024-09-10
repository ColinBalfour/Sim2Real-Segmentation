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
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the parameters
from loadParam import *


if os.path.exists(JOB_FOLDER):
    shutil.rmtree(JOB_FOLDER)
    print(f"deleted previous job folder from {JOB_FOLDER}")
os.mkdir(JOB_FOLDER)
os.mkdir(TRAINED_MDL_PATH)

# DATASET ---------------------------------------------------------------------------
datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device', device)

# Define the dataset size
# dataset = WindowDataset(DS_PATH)

# Split the dataset into train and validation
# dataset_size = len(dataset)

# train_size = int(0.9 * dataset_size)
# test_size = dataset_size - train_size
# trainset, valset = torch.utils.data.random_split(dataset, [train_size, test_size])

class MyTransform:
    def __init__(self):
        return
    def __call__(self, input, target):
        input = F.to_tensor(input)
        target = (F.to_tensor(target) * 255).int() #[:3, :, :]
        return (F.resize(input, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT), 
                F.resize(target, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT))

trainset = Cityscapes(DS_PATH, split='train', mode='fine', target_type='semantic', transforms=MyTransform())
valset = Cityscapes(DS_PATH, split='val', mode='fine', target_type='semantic', transforms=MyTransform())
    
trainLoader = torch.utils.data.DataLoader(trainset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
valLoader = torch.utils.data.DataLoader(valset, BATCH_SIZE, True, num_workers=NUM_WORKERS)


# Network and optimzer --------------------------------------------------------------
model = Network2(3, 34)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# model = fcn_resnet50(num_classes=3)
model = model.to(device)

# LOSS FUNCTION AND OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def shouldLog(batchcount=None):
    if batchcount==None:
        return LOG_WANDB=='true'
    else:
        return batchcount%LOG_BATCH_INTERVAL == 0 and LOG_WANDB=='true'

# INIT LOGGER
wandb.init(
    project=MODEL_NAME,
    name=str(JOB_ID),
    
    # track hyperparameters and run metadata
    config={
    "JOB_ID":JOB_ID,
    "learning_rate": LR,
    "batchsize": BATCH_SIZE,
    "dataset": DS_PATH,
    }
)

#  TRAIN ----------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epochstep):
    
    # dp('train started')
    model.train()
    epochloss = 0
    for batchcount, (rgb, label) in enumerate(dataloader):
        dp(' batch', batchcount)
        
        rgb = rgb.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        label[label < 0] = 26 # relabel license plate to car
        # print(torch.max(label))
        label = label.flatten(1, 2).long()

        pred = model(rgb)
        # print("shapes", pred.shape, label.shape)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        
        epochloss += loss.item()

        wandb.log({
            "epochstep": epochstep,
            "batch/loss/train": loss.item(),
                })
            
        if batchcount == 0: # only for the first batch every epoch
            wandb_images = []
            for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                combined_image_np = CombineImages(pred_single, label_single.unsqueeze(-1), rgb_single)

                # Create wandb.Image object and append to the list
                wandb_images.append(wandb.Image(combined_image_np))

            wandb.log(
            {
                "images/train": wandb_images,
            })
                    
    if shouldLog():
        wandb.log({
            "epoch/loss/train": epochloss,
                    })
    
# Define the val function
def val(dataloader, model, loss_fn, epochstep):
    model.eval()
    
    epochloss = 0
    with torch.no_grad():
        for batchcount, (rgb, label) in enumerate(dataloader):
            dp(' batch', batchcount)
            
            rgb = rgb.to(device)
            label = label.to(device)
            
            pred = model(rgb)
            label[label < 0] = 26 # relabel license plate to car
            # print(torch.max(label))
            label = label.flatten(1, 2).long()
            loss = loss_fn(pred, label)

            epochloss += loss.item()
        
            wandb.log({
                "batch/loss/": loss.item(),
                    })
            
            if batchcount == 0: # only for the first batch every epoch
                wandb_images = []
                for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                    combined_image_np = CombineImages(pred_single, label_single.unsqueeze(-1), rgb_single)
                    wandb_images.append(wandb.Image(combined_image_np))

                wandb.log(
                {
                    "images/val": wandb_images,
                })
            
    wandb.log({
        "epoch/loss/val": epochloss,
                })

# STORE ORIGINAL PARAMTERS
trainedMdlPath = TRAINED_MDL_PATH + f"test.pth"
torch.save(model.state_dict(), trainedMdlPath)

# SCRIPT ---------------------------------------------------------------------------------
epochs = 100

lossFn = nn.CrossEntropyLoss()

for eIndex in range(epochs):
    dp(f"Epoch {eIndex+1}\n")

    train(trainLoader, model, lossFn, optimizer, eIndex)
    val(valLoader, model, lossFn, eIndex)

    trainedMdlPath = TRAINED_MDL_PATH + f"{eIndex}.pth"
    torch.save(model.state_dict(), trainedMdlPath)