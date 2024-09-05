import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from dataaug import *
from loadParam import *
import pdb

import cv2
import numpy as np

import glob
import json


class WindowDataset(Dataset):
    def __init__(self, ds_path):
        # init code
        print("dataset init")

    def __len__(self):
        # Set the dataset size here
        return N

    def __getitem__(self, idx):
        # idx is from 0 to N-1

        # Open the RGB image and ground truth label

        # convert them to tensors

        # apply any transform (blur, noise...)

        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)

    rgb, label = dataset[0]


class kaggle_brain_tumor_ds:
    def __init__(self, ds_path, transform=None):
        self.ds_path = ds_path
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.load_data()

    def load_path(self):
        self.train_path = f"{self.ds_path}/train/"
        self.test_path = f"{self.ds_path}/test/"
        self.valid_path = f"{self.ds_path}/valid/"

        self.train = [
            image for image in os.listdir(self.train_path) if image[-3:] == "jpg"
        ]
        self.test = [
            image for image in os.listdir(self.test_path) if image[-3:] == "jpg"
        ]
        self.valid = [
            image for image in os.listdir(self.valid_path) if image[-3:] == "jpg"
        ]

        self.train_annotation = glob.glob(os.path.join(self.train_path, "*.json"))
        self.test_annotation = glob.glob(os.path.join(self.test_path, "*.json"))
        self.valid_annotation = glob.glob(os.path.join(self.valid_path, "*.json"))
        self.train_annotation = json.load(open(self.train_annotation[0]))
        self.test_annotation = json.load(open(self.test_annotation[0]))
        self.valid_annotation = json.load(open(self.valid_annotation[0]))

        self.create_mask("train", self.train_annotation, self.train_path)
        self.create_mask("test", self.test_annotation, self.test_path)
        self.create_mask("valid", self.valid_annotation, self.valid_path)

    def create_mask(self, mask_name, annotation, image_path):
        print("train masks")
        mask_dir = f"{image_path}/{mask_name}_masks/"
        os.makedirs(mask_dir, exist_ok=True)
        totalImages = len(self.annotation["images"])
        done = 0
        for img, ann in zip(self.annotation["images"], annotation["annotations"]):
            path = image_path + img["file_name"]
            mask_path = mask_dir + img["file_name"]
            # load image in open cv
            image = cv2.imread(path)
            segmentation = ann["segmentation"]
            segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [segmentation], color=(255, 255, 255))
            cv2.imwrite(mask_path, mask)
            done += 1
            print(f"train  {done} / {totalImages} ")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
