import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
# from dataaug import *
from loadParam import *
import pdb
import numpy as np
from torchvision.transforms import functional as F

class WindowDataset(Dataset):
    def __init__(self, ds_path):
        # init code
        print("dataset init")
        #image format is output_i.jpg
        # mask format is output_i.jpg
        self.image_dir = os.path.join(ds_path, 'images')
        self.mask_dir = os.path.join(ds_path, 'instance')
        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        # Set the dataset size here
        return self.num_images

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        if idx >= len(self):
            print(f"Index {idx} out of range, only {len(self)} items in dataset")
            raise IndexError
        
        # Open the RGB image and ground truth label
        rgb = Image.open(os.path.join(self.image_dir, f"output_{idx}.jpg"))
        label = Image.open(os.path.join(self.mask_dir, f"output_{idx}.jpg"))

        # convert them to tensors
        rgb = transforms.ToTensor()(rgb) # (3, H, W)
        label = F.rgb_to_grayscale(transforms.ToTensor()(label)) # (1, H, W)

        rgb = F.resize(rgb, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT)
        label = F.resize(label, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT)

        # apply any transform (blur, noise...)
        
        
        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)

    rgb, label = dataset[0]
