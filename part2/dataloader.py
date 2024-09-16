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
    def __init__(self, ds_path, instance=False):
        # init code
        print("dataset init")
        #image format is output_i.jpg
        # mask format is output_i.jpg
        self.image_dir = os.path.join(ds_path, 'images')
        self.mask_dir = os.path.join(ds_path, 'instance')
        self.num_images = len(os.listdir(self.image_dir))
        self.instance = instance

    def __len__(self):
        # Set the dataset size here
        return self.num_images

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        if idx >= len(self):
            print(f"Index {idx} out of range, only {len(self)} items in dataset")
            raise IndexError
        
        # Open the RGB image and ground truth label
        rgb = Image.open(os.path.join(self.image_dir, f"output_{idx}.png"))
        label = Image.open(os.path.join(self.mask_dir, f"output_{idx}.png"))
        
        # Convert the image to grayscale
        label = label.convert('L')

	# Convert grayscale to black and white
        #if self.instance:
        #    label = label.point(lambda x: 0 if x <= 30 else x, '1')
        #    label = label.point(lambda x: 1 if x > 30 and x <= 120 else x, '1')
        #    label = label.point(lambda x: 2 if x > 120 and x <=210 else x, '1')
        #    label = label.point(lambda x: 3 if x > 210 else x, '1')
        if not self.instance:
            label = label.point(lambda x: 255 if x >= 10 else 0, '1')
        
        #print('hi0', label.getextrema())
        # convert them to tensors
        rgb = transforms.ToTensor()(rgb) # (3, H, W)
        if not self.instance:
           label = F.to_tensor(label) # (1, H, W)
        else:
            label = F.pil_to_tensor(label)
            # Apply thresholds
            label[(label > 0) & (label <= 30)] = 0
            label[(label == 85)] = 3
            label[label == 170] = 2
            label[label == 255] = 1
            label[label > 3] = 0
        #print('hi', torch.max(label.squeeze()), label.shape)
        
        
        
        #print('hi2', torch.max(label))
        
        rgb = F.resize(rgb, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT)
        label = F.resize(label, [256, 256], interpolation= F.InterpolationMode.NEAREST_EXACT)

        # apply any transform (blur, noise...)
        resize = transforms.Resize((256, 256))
        
        i, j, th, tw = transforms.RandomCrop.get_params(rgb, output_size=torch.randint(50, 256, (2,)))
        crop = transforms.Lambda(lambda x: F.crop(x, i, j, th, tw))
        
        ret = transforms.RandomAffine.get_params((0, 0), (0.1, 0.1), (0.9, 2), None, (256, 256))
        homography = transforms.Lambda(lambda x: F.affine(x, *ret))
        
        blur = transforms.GaussianBlur(5)
        noise = transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.15)
        
        rgb = resize(crop(blur(noise(rgb))))
        label = resize(crop(label))
        #rgb = resize(homography(blur(noise(rgb))))
        #label = resize(homography(label))
        
        # print(rgb.shape, label.shape)
        
        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)

    rgb, label = dataset[0]
