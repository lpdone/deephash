# ADSH(AAAI2018)
# paper [Asymmetric Deep Supervised Hashing]
# https://cs.nju.edu.cn/lwj/paper/AAAI18_ADSH.pdf

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from config.GPUManager import GPUManager

torch.multiprocessing.set_sharing_strategy('file_system')

# auto select device
if torch.cuda.is_available():
    gpu = GPUManager().auto_choice()
    device = 'cuda:{0}'.format(gpu)
else:
    device = 'cpu'


def get_config():
    config = {
        "gamma": 200,
        "num_samples": 150,
        "max_iter": 150,
        "epoch": 3,
        "test_map": 10,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4}},
        "info": "[ADSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": ResNet,
        "dataset": "CUB200",
        "save_path": "save/ADSH",
        "device": device,
        "bit_list": [64, 48, 32, 16],
    }
    config = config_dataset(config)
    return config


def calc_sim(database_label, train_label):
    S = (database_label @ train_label.t() > 0).float()
    # soft constraint
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def train_val(config, bit):
    device = config["device"]
    num_samples = config["num_samples"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    # get database_labels
    clses = []
    for _, cls, _ in tqdm(dataset_loader):
        clses.append(cls)
    database_labels = torch.cat(clses).to(device).float()

    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.6 * config['epoch']),
                                                           int(0.8 * config['epoch'])],
                                               gamma=0.1)

    Best_mAP = 0

    V = torch.zeros((num_dataset, bit)).to(device)
    for iter in range(config["max_iter"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s,  training...." % (
            config["info"], iter + 1, config["max_iter"], current_time, bit, config["dataset"]), end="")

        net.train()

        # sampling and construct similarity matrix
        select_index = np.random.permutation(range(num_dataset))[0: num_samples]
        train_loader.dataset.imgs = np.array(dataset_loader.dataset.imgs, dtype=object)[select_index].tolist()
        sample_label = database_labels[select_index]

        Sim = calc_sim(sample_label, database_labels)
        U = torch.zeros((num_samples, bit)).to(device)

        train_loss = 0
        for epoch in range(config["epoch"]):
            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device).float()
                net.zero_grad()
                S = calc_sim(label, database_labels)
                u = net(image)
                u = u.tanh()
                U[ind, :] = u.data

                square_loss = (u @ V.t() - bit * S).pow(2)
                quantization_loss = config["gamma"] * (V[select_index[ind]] - u).pow(2)
                loss = (square_loss.sum() + quantization_loss.sum()) / (num_train * u.size(0))

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            scheduler.step()

        train_loss = train_loss / len(train_loader) / epoch
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        # learning binary codes: discrete coding
        barU = torch.zeros((num_dataset, bit)).to(device)
        barU[select_index, :] = U
        # calculate Q
        Q = -2 * bit * Sim.t() @ U - 2 * config["gamma"] * barU
        for k in range(bit):
            sel_ind = np.setdiff1d([ii for ii in range(bit)], k)
            V_ = V[:, sel_ind]
            U_ = U[:, sel_ind]
            Uk = U[:, k]
            Qk = Q[:, k]
            # formula 10
            V[:, k] = -(2 * V_ @ (U_.t() @ Uk) + Qk).sign()

        if (iter + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

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
                config["info"], iter + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
