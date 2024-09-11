import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple
import cv2

CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

classes = [
    CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]


def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)

def colorize(img):
    # print(img.shape)
    colorized_img = np.zeros((*img.shape, 3))

    # Map each class index to its corresponding color
    for city_class in classes:
        colorized_img[img == city_class.id] = city_class.color

    return colorized_img / 255

def CombineImages(pred, label, rgb):
    pred = pred.detach().squeeze()
    label = label.detach().cpu().numpy().squeeze()
    rgb = rgb.detach().cpu().numpy()

    # print(pred)
    # print(np.max(pred))
    num_classes = pred.shape[0]

    # new_pred = np.zeros((pred.shape[1], pred.shape[2]))
    # prob_map = np.zeros((pred.shape[1], pred.shape[2]))
    # for idx, y in enumerate(pred):
    #     new_pred[y > prob_map] = idx * 8
    #     prob_map[y > prob_map] = y[y > prob_map]

    # print(pred.shape)

    # probabilities = F.softmax(pred, dim=0)
    # predicted_classes = torch.argmax(probabilities, dim=0)
    # new_pred = colorize(predicted_classes.cpu().numpy()) #/ num_classes
    # label = colorize(label)  #/ num_classes
    
    # print(torch.max(pred))
    # print(torch.max(F.sigmoid(pred)))
    # print(F.sigmoid(pred).cpu().numpy().shape)
    # print(np.min(F.sigmoid(pred).cpu().numpy() * 255))

    new_pred = F.sigmoid(pred)
    new_pred[new_pred > 0.5] = 1
    new_pred[new_pred <= 0.5] = 0
    new_pred = cv2.cvtColor(new_pred.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)

    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    rgb = np.transpose(rgb, (1, 2, 0))

    # print(rgb.shape)
    # colormap = np.random.randint(0, 256, size=(num_classes, 3))
    # new_pred = colormap[predicted_classes.cpu().numpy()]
    # print('thing', pred.shape, label.shape)
    # new_pred = 0.299 * pred[0, :, :] + 0.587 * pred[1, :, :] + 0.114 * pred[2, :, :]
    # label = 0.299 * label[0, :, :] + 0.587 * label[1, :, :] + 0.114 * label[2, :, :]

    # new_pred = pred
    
    gray_array = 0.299 * rgb[0, :, :] + 0.587 * rgb[1, :, :] + 0.114 * rgb[2, :, :]

    # Concatenate images horizontally
    print(pred.shape, new_pred.shape, label.shape, rgb.shape, gray_array.shape)
    combined_image_np = np.concatenate((new_pred, label, rgb), axis=1)
    combined_image_np = (np.clip(combined_image_np, 0, 1)*255).astype(np.uint8)
    return combined_image_np