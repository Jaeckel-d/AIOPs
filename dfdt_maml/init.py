import copy
import argparse
import os
import datetime as dt
import torch
from torch import nn
from model import TC
import torch.optim
from data import DiskDataset
from torch.utils.data import DataLoader
from utils import Logger, check_dir

from multiprocessing import Pool, Queue


class MAML:
    def __init__(self, outer_lr, inner_lr, max_iters, episodic_csv_path,
                 model, inner_criterion, outer_criterion, device, logger):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_iters = max_iters
        self.meta_train_path = os.path.join(episodic_csv_path, "meta_train")
        self.meta_test_path = os.path.join(episodic_csv_path, "meta_test")
        self.meta_train_list = os.listdir(self.meta_train_path)
        self.meta_test_list = os.listdir(self.meta_test_path)
        assert len(self.meta_test_list) == len(self.meta_train_list)
        self.task_count = len(self.meta_train_list)
        self.model = model
        self.device = device

        self.model_name = "{}_maml".format(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d-%H_%M_%S'))
        self.inner_criterion = inner_criterion
        self.outer_criterion = outer_criterion
        self.logger = logger
        self.log_inter = 10

        self.logger.log("MAML initailized, outer_lr:{}, lr_beta:{}, tasks:{}, max iter: {}, device: {}"
                        .format(outer_lr, inner_lr, self.task_count, max_iters, self.device))

    def optim(self):
        self.model = self.model.to(self.device)
        # outer_optim = torch.optim.Adam(lr=self.outer_lr, params=self.model.parameters(), betas=(0.9, 0.98))
        outer_optim = torch.optim.SGD(lr=self.outer_lr, params=self.model.parameters())
        # outer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(outer_optim, self.max_iters)
        outer_scheduler = None
        for it in range(self.max_iters):
            print("---------Iter {} begin ---------".format(it))

            outer_loss = torch.tensor(.0).to(self.device)

            for task_id in range(self.task_count):

                model_copy = copy.deepcopy(self.model)
                model_copy.load_state_dict(self.model.state_dict())

                meta_train_ds = DiskDataset(
                    csv_path=os.path.join(self.meta_train_path, "task_{}_train.csv".format(task_id)),
                    data_path="./data/preprocessed",
                )
                meta_train_dl = DataLoader(meta_train_ds, batch_size=1, shuffle=True)

                inner_optim = torch.optim.SGD(lr=self.inner_lr, params=model_copy.parameters())
                inner_loss = torch.tensor(.0).to(self.device)
                for src, tgt in meta_train_dl:
                    src = src.float().to(self.device)
                    tgt = tgt.float().to(self.device)
                    out = model_copy(src)

                    inner_loss = self.inner_criterion(out, tgt)
                    inner_loss.backward()
                print("(Iter {})(Task {})Inner loss: {}".format(it, task_id, inner_loss))
                inner_optim.step()
                inner_optim.zero_grad()

                # self.model.load_state_dict(model_copy.state_dict())
                self.model = copy.deepcopy(model_copy)

                meta_test_ds = DiskDataset(
                    csv_path=os.path.join(self.meta_test_path, "task_{}_test.csv".format(task_id)),
                    data_path="./data/preprocessed"
                )
                meta_test_dl = DataLoader(meta_test_ds, batch_size=1, shuffle=True)
                test_loss = torch.tensor(.0).to(self.device)

                for src, tgt in meta_test_dl:
                    src = src.float().to(self.device)
                    tgt = tgt.float().to(self.device)
                    out = self.model(src)
                    loss = self.inner_criterion(out, tgt)
                    test_loss += loss
                outer_loss += test_loss
            if it % self.log_inter == 0:
                self.logger.log("(Iter {}) Outer loss: {}".format(it, outer_loss))
            if it == 0:
                e1_name = "{}_epoch_1.pt".format(self.model_name)
                self.save_model("./run/init/{}".format(e1_name))
                self.logger.log("Model epoch 1 saved at: {}".format(e1_name))
            if it == int(self.max_iters / 2):
                eh_name = "{}_epoch_{}.pt".format(self.model_name, int(self.max_iters/2))
                self.save_model("./run/init/{}".format(eh_name))
                self.logger.log("Model epoch half saved at: {}".format(eh_name))
            outer_loss.backward()
            outer_optim.step()
            outer_optim.zero_grad()
            if outer_scheduler:
                outer_scheduler.step()
        self.save_model("./run/init/{}.pt".format(self.model_name))
        self.logger.log("Model saved as name: {}".format(self.model_name))

    def save_model(self, path):
        torch.save(self.model, path)


parser = argparse.ArgumentParser(description="FOMAML",)
parser.add_argument('--name', type=str, help='log name', default='meta_init')
parser.add_argument('--inner-lr', type=float, default=0.01)
parser.add_argument('--outer-lr', type=float, default=0.01)
parser.add_argument('--max-iter', type=int, default=300)


def run_fomaml(exp_name, outer_lr, inner_lr, max_iter=300):

    log_dir = "./run/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = "{}/{}.log".format(log_dir, exp_name)

    transformer_classifer = TC(
        d_model=12,
        nhead=1,
        num_layers=3,
        dim_feedforward=64,
        hidden_size=32,
        # out_features=1,
    )

    maml = MAML(outer_lr=outer_lr, inner_lr=inner_lr, max_iters=max_iter, episodic_csv_path="data/episodic",
                model=transformer_classifer,
                inner_criterion=nn.BCELoss(), outer_criterion=nn.BCELoss(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                logger=Logger(log_path))
    maml.optim()


def fomaml_mt(pool_size=1):

    outer_lrs = [0.1, 0.01, 0.001]
    inner_lrs = [0.001, 0.01, 0.1]
    cnt = 0
    mt_pool = Pool(pool_size)
    for olr in outer_lrs:
        for ilr in inner_lrs:
            cnt += 1
            exp_name = "config{}".format(cnt)
            mt_pool.apply_async(run_fomaml, args=(exp_name, olr, ilr))

    mt_pool.close()
    mt_pool.join()


if __name__ == "__main__":

    fomaml_mt(pool_size=3)

    # arg = parser.parse_args()
    # name = arg.name
    #
    # lr_inner = arg.inner_lr
    # lr_outer = arg.outer_lr
    # max_iter = arg.max_iter
    #
    # log_dir = "./run/log"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_path = "{}/{}.log".format(log_dir, name)
    #
    # transformer_classifer = TC(
    #     d_model=12,
    #     nhead=1,
    #     num_layers=3,
    #     dim_feedforward=64,
    #     hidden_size=32,
    #     # out_features=1,
    # )
    #
    # maml = MAML(outer_lr=lr_outer, inner_lr=lr_inner, max_iters=max_iter, episodic_csv_path="data/episodic",
    #             model=transformer_classifer,
    #             inner_criterion=nn.BCELoss(), outer_criterion=nn.BCELoss(),
    #             device="cuda" if torch.cuda.is_available() else "cpu",
    #             logger=Logger(log_path))
    #
    # maml.optim()
