import torch
from torch import nn
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import TransformerClr, TC
from data import DiskDataset
import matplotlib.pyplot as plt
import numpy as np


class ModelWrapper:
    def __init__(self, model, state_dict=None, dataset=None,
                 criterion=nn.BCELoss(), device="cpu"):
        assert model is not None
        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=8 ,shuffle=True)
        self.criterion = criterion
        self.device = device

        if state_dict is not None:
            assert model is not None
            self.model.load_state_dict(state_dict)

    def eval(self, state_dict=None):
        assert self.dataloader is not None
        assert self.model is not None
        assert self.criterion is not None
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model = self.model.to(self.device)
        val_total, val_loss = len(self.dataloader), .0

        for src, tgt in self.dataloader:
            src = src.float().to(self.device)
            tgt = tgt.float().to(self.device)

            out = self.model(src)
            loss = self.criterion(out, tgt)

            val_loss += loss.item()

        return val_loss / val_total


def state_dict_sum_coef(sd_0, sd_1, alpha):
    sd = OrderedDict()
    for k in sd_0.keys():
        sd[k] = sd_0[k] + alpha * sd_1[k]
    return sd


def state_dict_mult_n(sd_0, alpha):
    sd = OrderedDict()
    for k in sd_0.keys():
        sd[k] = alpha * sd_0[k]
    return sd


def state_dict_diff(sd_0: OrderedDict, sd_1: OrderedDict):
    sd_diff = OrderedDict()
    for k in sd_0.keys():
        sd_diff[k] = sd_1[k] - sd_0[k] + 1e-9
    return sd_diff


def interpolate_list(arr, size):
    res = np.zeros((2 * size - 1))
    for i in range(0, size):
        res[2 * i] = arr[i]

    for i in range(1, 2 * size - 1, 2):
        res[i] = (res[i - 1] + res[i + 1]) / 2

    return res


def interpolate_array(arr, size):
    res = np.zeros((2 * size - 1, 2 * size - 1))
    for i in range(0, size):
        for j in range(0, size):
            res[2 * i, 2 * j] = arr[i, j]

    for i in range(1, 2 * size - 1, 2):
        for j in range(0, size):
            res[i, 2 * j] = (res[i - 1, 2 * j] + res[i + 1, 2 * j]) / 2

    for i in range(0, size):
        for j in range(1, 2 * size - 1, 2):
            res[2 * i, j] = (res[2 * i, j - 1] + res[2 * i, j + 1]) / 2

    for i in range(1, 2 * size - 1, 2):
        for j in range(1, 2 * size - 1, 2):
            res[i, j] = (res[i - 1, j - 1] +
                         res[i - 1, j + 1] +
                         res[i + 1, j - 1] +
                         res[i + 1, j + 1]) / 4

    return res


def interpolation_2D(model_paths: List, alpha: List, beta: List, n: int):
    model_0 = torch.load(model_paths[0], map_location="cpu")
    model_1 = torch.load(model_paths[1], map_location="cpu")
    model_2 = torch.load(model_paths[2], map_location="cpu")

    sd_0 = model_0.state_dict()
    sd_1 = model_1.state_dict()
    sd_2 = model_2.state_dict()

    delta_01 = state_dict_diff(sd_0, sd_1)
    delta_02 = state_dict_diff(sd_0, sd_2)
    delta_01_norm = torch.sqrt(sum([torch.sum(torch.square(t)) for t in delta_01.values()]))
    delta_02_norm = torch.sqrt(sum([torch.sum(torch.square(t)) for t in delta_02.values()]))
    delta_02_scaled = state_dict_mult_n(delta_02, delta_01_norm / delta_02_norm)

    x = [alpha[0] + (alpha[1] - alpha[0]) * i / (n - 1.0) for i in range(0, n)]
    y = [beta[0] + (beta[1] - beta[0]) * i / (n - 1.0) for i in range(0, n)]
    z = []

    mw = ModelWrapper(model=TC(d_model=12, nhead=1, num_layers=3, dim_feedforward=64, hidden_size=32,),
                      dataset=DiskDataset(csv_path="./data/val_sample.csv", data_path="data/preprocessed/"))

    for i in range(0, n):
        tmp = []
        for j in range(0, n):
            print("Interpolation: ({}, {})".format(i, j))
            sd = state_dict_sum_coef(state_dict_sum_coef(sd_0, delta_01, x[i]), delta_02_scaled, y[j])
            tmp.append(mw.eval(sd))
        z.append(tmp)

    return x, y, z


def plot_landscape(x, y, z, n, save_name="loss_landscape.png"):
    fig, ax = plt.subplots()

    # z_loss = []

    z = interpolate_array(np.array(z), n)
    x_ip = interpolate_list(np.array(x), n)
    y_ip = interpolate_list(np.array(y), n)

    X, Y = np.meshgrid(x_ip, y_ip)

    cp = ax.contourf(X, Y, z, levels=15)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('2-D interpolation initial parameters and so-called minimizer')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.savefig(save_name, dpi=200)
    # plt.show()


if __name__ == "__main__":

    tc_model_paths = [
        "run/2023-05-11-13_30_40_ckpt_epoch_1.pt",
        "run/2023-05-11-13_30_40_ckpt_epoch_250.pt",
        "run/2023-05-11-13_30_40_ckpt.pt",
    ]

    tc_model_maml_paths = [
        "run/2023-05-16-10_45_38_ckpt.pt",
        "run/2023-05-16-10_45_38_ckpt_epoch_250.pt",
        "run/2023-05-16-10_45_38_ckpt_epoch_1.pt",
    ]

    tc_init_maml_paths = [
        "run/init/2023-05-22-05_19_53_maml_epoch_1.pt",
        "run/init/2023-05-22-05_19_53_maml_epoch_150.pt",
        "run/init/2023-05-22-05_19_53_maml.pt",
    ]

    png_name = "plots/{}_loss_landscape.png".format(tc_init_maml_paths[2].split("/")[1].split(".")[0])
    print(png_name)

    N = 100
    drange = [-100, 100]
    _x, _y, _z = interpolation_2D(tc_init_maml_paths, alpha=drange, beta=drange, n=N)
    plot_landscape(_x, _y, _z, N, png_name)
