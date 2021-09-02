import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
# import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import imageio

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


class CarsDataset(torch.utils.data.Dataset):

    def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)
            for c in self.car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[idx][-2][0][0]
        car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        assert car_class < 196
        
        if self.transform:
            image = self.transform(image)

        # return image, car_class, img_name
        return image, car_class, idx

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()


class Food101(Dataset):

    def __init__(self, data_dir, is_train=True, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = os.path.join(data_dir, "images")
        self.meta_dir = os.path.join(data_dir, "meta")
        self.classes_dir = os.path.join(self.meta_dir, "classes.txt")
        with open(self.classes_dir, "r") as f:
            classes_names = [name.strip() for name in f.readlines()]
        self.class2id = {}
        for idx, name in enumerate(classes_names):
            self.class2id[name] = idx
        if is_train:
            with open(os.path.join(self.meta_dir, "train.txt"), "r") as f:
                self.img_names = [name.strip() for name in f.readlines()]
        else:
            with open(os.path.join(self.meta_dir, "test.txt"), "r") as f:
                self.img_names = [name.strip() for name in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        food_class = img_name.split("/")[-2]
        class_id = self.class2id[food_class]
        assert class_id < 101
        if self.transform:
            image = self.transform(image)
        return image, class_id, idx
