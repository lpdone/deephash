# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval]
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf

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
        "alpha": 0.1,
        "optimizer": {'type': optim.Adam, 'optim_params': {'lr': 1e-4}},
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "net": ResNet,
        "dataset": 'StanfordDogs',
        "epoch": 250,
        "test_map": 15,
        "save_path": os.path.join(Path.project_root, 'save/DSH'),
        "device": device,
        "bit_list": [64, 48, 32, 16],
    }
    config = config_dataset(config)
    return config


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.sign()).abs().mean()

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

    criterion = DSHLoss(config, bit)

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
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/CUB200-16-0.21070122289449728-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/StanfordDogs-16-0.16951685077614798-model.pth'
    elif bit == 32:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/CUB200-32-0.5665556321206521-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/StanfordDogs-32-0.25494072263626183-model.pth'
    elif bit == 48:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/CUB200-48-0.7272969559119643-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/StanfordDogs-48-0.49016968632315483-model.pth'
    elif bit == 64:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/CUB200-64-0.7367623582608954-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/DSH/StanfordDogs-64-0.6494740968088117-model.pth'

    net.load_state_dict(torch.load(pth), strict=True)
    net.to(device)

    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    print('DSH %d mAP: %.4f' % (bit, mAP * 100))

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
