import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from pycocotools.coco import COCO
import random


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        use_same = random.choice([True, False])
        if use_same:
            image_2 = np.array(image)
        else:
            rand_range = (list(range(0, index)) + 
                          list(range(index + 1, len(self.images))))
            img_path_2 = os.path.join(self.image_dir, 
                                      self.images[random.choice(rand_range)])
            image_2 = np.array(Image.open(img_path_2).convert('RGB'), 
                               dtype=np.float32)                        
        augmentations = self.transform(image=image)
        augmentations_2 = self.transform(image=image_2)
        image = augmentations['image']
        image_2 = augmentations_2['image']

        return image, image_2, int(use_same)
