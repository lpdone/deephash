from PIL import Image
import os

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CarsDataset, Food101


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


def get_loader(dataset, data_root):

    transform_train = image_transform(256, 224, "train_set")
    transform_test = image_transform(256, 224, "test_set")
    
    if dataset == 'car':
        trainset = CarsDataset(os.path.join(data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(data_root,'cars_train'),
                            os.path.join(data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transform_train
                            )
        testset = CarsDataset(os.path.join(data_root,'devkit/cars_test_annos_withlabels.mat'),
                            os.path.join(data_root,'cars_test'),
                            os.path.join(data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transform_test
                            )
    elif dataset == 'food101':
        trainset = Food101(data_root, True, transform=transform_train)
        testset = Food101(data_root, False, transform=transform_test)



    train_sampler = RandomSampler(trainset)
    gallery_sampler = SequentialSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
    gallery_loader = DataLoader(trainset,
                              sampler=gallery_sampler,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=64,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader, gallery_loader, len(trainset), len(testset), len(trainset)
