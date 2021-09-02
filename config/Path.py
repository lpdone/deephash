import torch
import numpy as np
import random
import os
from config.ServerManager import ServerManager

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)


class Path(object):
    root = ServerManager().get_root()
    project_root = os.path.join(root, 'Project/DeepHash')
    data_root = os.path.join(root, 'Project/Dataset')
