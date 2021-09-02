# DPN(IJCAI2020)
# paper [Deep Polarized Network for Supervised Learning of Accurate Binary Hashing Codes]
# https://www.ijcai.org/Proceedings/2020/115

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random

from config.GPUManager import GPUManager
from utils.data_utils import get_loader
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# auto select device
if torch.cuda.is_available():
    gpu = GPUManager().auto_choice()
    device = 'cuda:{0}'.format(gpu)
else:
    device = 'cpu'


def get_config():
    config = {
        "m": 1,
        "p": 0.5,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4}},
        "info": "[DPN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": ResNet,
        # "dataset": 'food101',
        # "data_root": '/data/ludi/datasets/food-101',
        # "epoch": 100,
        "dataset": 'car',
        "data_root": '/data/ludi/datasets/StanfordCars',
        "epoch": 300,
        "test_map": 5,
        "save_path": './output/DPN',
        "device": device,
        "bit_list": [64, 48, 32, 16],
        "topK": -1,
        "n_class": 196,
    }
    # config = config_dataset(config)
    return config


class DPNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPNLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.target_vectors = self.get_target_vectors(config["n_class"], bit, config["p"]).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.m = config["m"]
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        if "-T" in config["info"]:
            # Ternary Assignment
            u = (u.abs() > self.m).float() * u.sign()

        t = self.label2center(y)
        polarization_loss = (self.m - u * t).clamp(0).mean()

        return polarization_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.target_vectors[y.argmax(axis=1)]
        else:
            # for multi label, use the same strategy as CSQ
            center_sum = y @ self.target_vectors
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # Random Assignments of Target Vectors
    def get_target_vectors(self, n_class, bit, p=0.5):
        target_vectors = torch.zeros(n_class, bit)
        for k in range(20):
            for index in range(n_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                target_vectors[index] = ones
        return target_vectors

    # Adaptive Updating
    def update_target_vectors(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.target_vectors = (self.Y.t() @ self.U).sign()


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_loader(config["dataset"], config["data_root"])
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.6 * config['epoch']),
                                                           int(0.8 * config['epoch'])],
                                               gamma=0.1)

    criterion = DPNLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = torch.eye(config["n_class"])[label].to(device)
            # label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        if "-A" in config["info"]:
            criterion.update_target_vectors()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device, n_class=config["n_class"])

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device, n_class=config["n_class"])

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"],
                                         config["dataset"] + "-" + str(bit) + "-" + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"],
                                            config["dataset"] + "-" + str(bit) + "-" + str(mAP) + "-" + "model.pth"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


def test(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit)

    pth = None
    if bit == 16:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/CUB200-16-0.7383280741265901-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/StanfordDogs-16-0.7397652204770622-model.pth'
    elif bit == 32:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/CUB200-32-0.7518075085230349-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/StanfordDogs-32-0.7551183236910851-model.pth'
    elif bit == 48:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/CUB200-48-0.778538113673046-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/StanfordDogs-48-0.7653640453505592-model.pth'
    elif bit == 64:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/CUB200-64-0.7824654071678129-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DPN/StanfordDogs-64-0.7790103495036129-model.pth'

    net.load_state_dict(torch.load(pth), strict=True)
    net.to(device)

    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    print('DPN %d mAP: %.4f' % (bit, mAP * 100))

    P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
    P_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'P.npy')
    R_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'R.npy')
    np.save(P_save, P)
    np.save(R_save, R)


if __name__ == "__main__":
    config = get_config()
    print(config)
    # for bit in config["bit_list"]:
    #     # train_val(config, bit)
    #     test(config, bit)
    bit = int(sys.argv[1])
    train_val(config, bit)
