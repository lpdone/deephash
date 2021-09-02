# DCH(CVPR2018)
# paper [Deep Cauchy Hashing for Hamming Space Retrieval]
# http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf

from utils.tools import *
from network import *
from utils.data_utils import get_loader

import os
import torch
import torch.optim as optim
import time
import numpy as np
import sys

from config.GPUManager import GPUManager
from config.Path import Path

torch.multiprocessing.set_sharing_strategy('file_system')

# auto select device
if torch.cuda.is_available():
    gpu = GPUManager().auto_choice()
    device = 'cuda:{0}'.format(gpu)
else:
    device = 'cpu'


def get_config():
    config = {
        "gamma": 20.0,
        "lambda": 0.1,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4}},
        "info": "[DCH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": ResNet,
        "dataset": 'food101',
        "data_root": '/data/ludi/datasets/food-101',
        "epoch": 100,
        "test_map": 5,
        "save_path": './output/DCH',
        "device": device,
        "bit_list": [64, 48, 32, 16],
        "topK": -1,
        "n_class": 101,
    }
    # config = config_dataset(config)
    return config


class DCHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DCHLoss, self).__init__()
        self.gamma = config["gamma"]
        self.lambda1 = config["lambda"]
        self.K = bit
        self.one = torch.ones((config["batch_size"], bit)).to(config["device"])

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind, config):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.lambda1 * quantization_loss.mean()

        return loss


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

    criterion = DCHLoss(config, bit)

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
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/CUB200-16-0.6713663061546133-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/StanfordDogs-16-0.7433241240947656-model.pth'
    elif bit == 32:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/CUB200-32-0.7123360053082982-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/StanfordDogs-32-0.7547498673304844-model.pth'
    elif bit == 48:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/CUB200-48-0.7284993032785975-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/StanfordDogs-48-0.7620897251642084-model.pth'
    elif bit == 64:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/CUB200-64-0.7522378490170428-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DCH/StanfordDogs-64-0.7909404930787411-model.pth'

    net.load_state_dict(torch.load(pth), strict=True)
    net.to(device)

    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    print('DCH %d mAP: %.4f' % (bit, mAP * 100))

    P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
    P_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'P.npy')
    R_save = os.path.join(config['save_path'], config['dataset'] + '-' + str(bit) + '-' + 'R.npy')
    np.save(P_save, P)
    np.save(R_save, R)


if __name__ == "__main__":
    config = get_config()
    print(config)
    # for bit in config["bit_list"]:
    bit = int(sys.argv[1])
    train_val(config, bit)
        # train_val(config, bit)
        # test(config, bit)
