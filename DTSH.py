# DTSH(ACCV2016)
# paper [Deep Supervised Hashing with Triplet Labels]
# https://arxiv.org/abs/1612.03900
import pathlib

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
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
        "alpha": 5,
        "lambda": 1,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4}},
        "info": "[DTSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "net": ResNet,
        "dataset": 'StanfordDogs',
        "epoch": 150,
        "test_map": 15,
        "save_path": os.path.join(Path.project_root, 'save/DTSH'),
        "device": device,
        "bit_list": [64, 48, 32, 16],
    }
    config = config_dataset(config)
    return config


class DTSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y, ind, config):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - config["alpha"]).clamp(min=-100,
                                                                                                             max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = config["lambda"] * (u - u.sign()).pow(2).mean()

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.6 * config['epoch']),
                                                           int(0.8 * config['epoch'])],
                                               gamma=0.1)

    criterion = DTSHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
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
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/CUB200-16-0.6951923091461775-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/StanfordDogs-16-0.6702679291421533-model.pth'
    elif bit == 32:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/CUB200-32-0.7187332263351274-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/StanfordDogs-32-0.7054800552708603-model.pth'
    elif bit == 48:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/CUB200-48-0.7273594185891489-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/StanfordDogs-48-0.7240635744844631-model.pth'
    elif bit == 64:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/CUB200-64-0.7366411844552587-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DTSH/StanfordDogs-64-0.738685209251828-model.pth'

    net.load_state_dict(torch.load(pth), strict=True)
    net.to(device)

    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    print('DTSH %d mAP: %.4f' % (bit, mAP * 100))

    P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
    P_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'P.npy')
    R_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'R.npy')
    np.save(P_save, P)
    np.save(R_save, R)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        # train_val(config, bit)
        test(config, bit)
