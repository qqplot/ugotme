import json
import os

import albumentations
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensorV2 as ToTensor
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import pickle

def load_corruption(path):
    data = np.load(path)
    return np.array(np.array_split(data, 5))


class CIFARDatasetAdapt(Dataset):

    base_folder = "cifar-10-batches-py"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(self, split, root_dir):
        
        self.root = root_dir
        self._X = []
        self._y = []
        if split == 'train':
            # now load the picked numpy arrays
            for file_name, checksum in self.train_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    self._X.append(entry["data"])
                    if "labels" in entry:
                        self._y.extend(entry["labels"])
                    else:
                        self._y.extend(entry["fine_labels"])

            self._X = np.vstack(self._X).reshape(-1, 3, 32, 32)
            self._X = self._X.transpose((0, 2, 3, 1))  # convert to HWC
            self.n_groups = 1
        if split == 'val':
            # now load the picked numpy arrays
            for file_name, checksum in self.test_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    self._X.append(entry["data"])
                    if "labels" in entry:
                        self._y.extend(entry["labels"])
                    else:
                        self._y.extend(entry["fine_labels"])

            self._X = np.vstack(self._X).reshape(-1, 3, 32, 32)
            self._X = self._X.transpose((0, 2, 3, 1))  # convert to HWC
            self.n_groups = 1
        if split == 'test':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C/'
            corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                           'defocus_blur', 'glass_blur', 'motion_blur',
                           'zoom_blur', 'snow', 'frost', 
                           'fog', 'brightness', 'contrast', 
                           'elastic_transform', 'pixelate', 'jpeg_compression']            
            data = [load_corruption(self.root_dir / (corruption + '.npy'))[-1] for corruption in corruptions]
            self._X = np.concatenate(data, axis=0)
            self.n_groups = len(corruptions)

        print("loading cifar-10")

        self.groups = list(range(self.n_groups))
        self.image_shape = (3, 32, 32)
        self.num_classes = 10        

        if split == 'test':
            self._X = self._X.reshape((-1, 32, 32, 3))
            n_images = 10000
            self._y = np.load(self.root_dir / 'labels.npy')[:n_images]
            self._y = np.tile(self._y, self.n_groups)
            self.group_ids = np.array([[i]*n_images for i in range(self.n_groups)]).flatten()
        else:
            n_images = len(self._X)
            self.group_ids = np.array([[i]*n_images for i in range(self.n_groups)]).flatten()

        print("loaded")

        self.transform = get_transform()
        print("split: ", split)
        print("n groups: ", self.n_groups)
        print("Dataset size: ", len(self._y))

    def __len__(self):
        return len(self._X)
    
    def __getitem__(self, index):
        x = self.transform(**{'image': self._X[index]})['image']
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)

        return x, y, g


def get_transform():
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                     		 std=[0.229, 0.224, 0.225], max_pixel_value=255,
                                 p=1.0, always_apply=True),
        ToTensor()
    ])
    return transform
