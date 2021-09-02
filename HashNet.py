# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation]
# http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf

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
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4}},
        "info": "[HashNet]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "net": ResNet,
        "dataset": 'StanfordDogs',
        "epoch": 150,
        "test_map": 15,
        "save_path": os.path.join(Path.project_root, 'save/HashNet'),
        "device": device,
        "bit_list": [64, 48, 32, 16],
    }
    config = config_dataset(config)
    return config


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


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

    criterion = HashNetLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")

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
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/CUB200-16-0.11977254770559809-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/StanfordDogs-16-0.18776919661986588-model.pth'
    elif bit == 32:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/CUB200-32-0.44802495373997514-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/StanfordDogs-32-0.5935443068899603-model.pth'
    elif bit == 48:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/CUB200-48-0.49786718470481245-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/StanfordDogs-48-0.626879472644662-model.pth'
    elif bit == 64:
        if config['dataset'] == 'CUB200':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/CUB200-64-0.5429848600112883-model.pth'
        elif config['dataset'] == 'StanfordDogs':
            pth = '/data/zengziyun/Project/DeepHash/save/HashNet/StanfordDogs-64-0.6348251813036061-model.pth'

    net.load_state_dict(torch.load(pth), strict=True)
    net.to(device)

    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    print('HashNet %d mAP: %.4f' % (bit, mAP * 100))

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
