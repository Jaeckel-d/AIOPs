import copy
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime as dt
from utils import move_to_device
from multiprocessing import Pool, Queue
from utils import Logger
from multiprocessing import Pool, Queue
from model import AttentionPool, AttentionLSTM, GateAttention
from aiop_dataset import MetaAIOPS


class MAML(object):
    def __init__(self,
                 outer_lr, inner_lr, epochs, train_dataloader, test_dataloader,
                 batch,
                 model, inner_criterion, outer_criterion,
                 device, logger,
                 ):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.epochs = epochs
        self.batch = batch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model_name = "{}_maml".format(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d-%H_%M_%S'))
        self.tasks_num = len(self.train_dataloader)

        self.model = model
        self.device = device

        self.outer_criterion = outer_criterion
        self.inner_criterion = inner_criterion
        self.logger = logger
        self.log_inter = 1

        self.logger.log("MAML initailized, outer_lr:{}, lr_beta:{}, tasks:{}, max iter: {}, device: {}"
                        .format(outer_lr, inner_lr, self.tasks_num, epochs, self.device))

    def optim(self):
        self.model = self.model.to(self.device)
        outer_optim = torch.optim.SGD(lr=self.outer_lr, params=self.model.parameters())
        outer_scheduler = None

        for epoch in range(self.epochs):
            print("---------Epoch {} begin ---------".format(epoch))

            outer_loss = torch.tensor(.0).to(self.device)

            for task in range(self.tasks_num):
                model_copy = copy.deepcopy(self.model)
                model_copy.load_state_dict(self.model.state_dict())

                x_sup, x_que, y_sup, y_que = next(iter(self.train_dataloader))

                inner_optim = torch.optim.SGD(lr=self.inner_lr, params=model_copy.parameters())
                inner_loss = torch.tensor(.0).to(self.device)

                x_sup, x_que = move_to_device(x_sup, self.device), \
                    move_to_device(x_que, self.device)
                y_sup, y_que = y_sup.to(self.device), y_que.to(self.device)

                for i, src in enumerate(x_sup):
                    tgt = y_sup[:, i]
                    out = model_copy(src)
                    inner_loss = self.inner_criterion(out, tgt)
                    inner_loss.backward()
                print("(Iter {})(Task {})Inner loss: {}".format(epoch, task * self.batch, inner_loss))
                inner_optim.step()
                inner_optim.zero_grad()

                self.model = copy.deepcopy(model_copy)

                test_loss = torch.tensor(.0).to(self.device)

                test_num = 0

                for i, src in enumerate(x_que):
                    tgt = y_que[:, i]
                    out = model_copy(src)
                    loss = self.inner_criterion(out, tgt)
                    test_loss += loss
                    test_num += 1
                outer_loss += test_loss

                # outer_loss /= test_num

            if epoch % self.log_inter == 0:
                self.logger.log("(Iter {}) Outer loss: {}".format(epoch, outer_loss))
            if epoch == 0:
                e1_name = "{}_epoch_1.pt".format(self.model_name)
                self.save_model("./run/init/{}".format(e1_name))
                self.logger.log("Model epoch 1 saved at: {}".format(e1_name))
            if epoch == int(self.epochs / 2):
                eh_name = "{}_epoch_{}.pt".format(self.model_name, int(self.epochs / 2))
                self.save_model("./run/init/{}".format(eh_name))
                self.logger.log("Model epoch half saved at: {}".format(eh_name))
            outer_loss.backward()
            outer_optim.step()
            outer_optim.zero_grad()
            if outer_scheduler:
                outer_scheduler.step()
        self.save_model("./run/init/{}.pt".format(self.model_name))
        self.logger.log("Model saved as name: {}".format(self.model_name))

    def finetune(self):
        self.model = self.model.to(self.device)
        outer_optim = torch.optim.SGD(lr=self.outer_lr, params=self.model.parameters())
        outer_scheduler = None


        model_copy = copy.deepcopy(self.model)
        model_copy.load_state_dict(self.model.state_dict())

        x_sup, x_que, y_sup, y_que = next(iter(self.train_dataloader))

        inner_optim = torch.optim.SGD(lr=self.inner_lr, params=model_copy.parameters())
        inner_loss = torch.tensor(.0).to(self.device)

        x_sup, x_que = move_to_device(x_sup, self.device), \
            move_to_device(x_que, self.device)
        y_sup, y_que = y_sup.to(self.device), y_que.to(self.device)

        for i, src in enumerate(x_sup):
            tgt = y_sup[:, i]
            out = model_copy(src)
            inner_loss = self.inner_criterion(out, tgt)
            inner_loss.backward()
        inner_optim.step()
        inner_optim.zero_grad()

        self.model = copy.deepcopy(model_copy)
        test_loss = torch.tensor(.0).to(self.device)

        test_num = 0

        for i, src in enumerate(x_que):
            tgt = y_que[:, i]
            out = model_copy(src)


    def save_model(self, path):
        torch.save(self.model, path)


parser = argparse.ArgumentParser(description="FOMAML", )
parser.add_argument('--name', type=str, help='log name', default='meta_init')
parser.add_argument('--inner-lr', type=float, default=0.01)
parser.add_argument('--outer-lr', type=float, default=0.01)
parser.add_argument('--max-iter', type=int, default=300)


def run_fomaml(exp_name, outer_lr, inner_lr, max_iter=300):
    log_dir = "./run/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = "{}/{}.log".format(log_dir, exp_name)

    model = AttentionLSTM()
    # aiops_train = MetaAIOPS(root_path="./aliyunwei/tmp_data", split="train")
    aiops_train = DataLoader(MetaAIOPS(root_path="./aliyunwei/tmp_data", split="train"), batch_size=4)

    maml = MAML(outer_lr=outer_lr, inner_lr=inner_lr, epochs=max_iter, train_dataloader=aiops_train,
                test_dataloader=None,
                batch=4,
                model=model,
                inner_criterion=nn.CrossEntropyLoss(), outer_criterion=nn.CrossEntropyLoss(),
                device="cuda:2" if torch.cuda.is_available() else "cpu",
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
    # fomaml_mt(pool_size=3)

    run_fomaml(exp_name="test_outer0.01_inner0.001", outer_lr=0.01, inner_lr=0.001)

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
