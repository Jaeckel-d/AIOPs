import datetime as dt
import os
import time
import wandb
import torch.optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, tensor, Tensor
# from torch.utils.tensorboard import SummaryWriter
from model import TC, FocalLoss, LSTMPredictor, TransformerClr
from data import DiskDataset

import matplotlib.pyplot as plt

train_csv_path = './data/train_sample.csv'
test_csv_path = './data/test_sample.csv'
val_csv_path = './data/val_sample.csv'
disk_data_path = "data/preprocessed/"
THRESHOLD = 0.8


class Solver:
    def __init__(self, device,
                 batch_size, epochs, lr,
                 train_dataloader, test_dataloader, val_dataloader,
                 model, optimizer, criterion, lr_schedular=None, grad_acc_step=3,
                 use_wandb=False, runtime_path="./run/", do_val=False, load_pretrain=False,
                 save=False, log_inter=50, ailyn_strat=False,
                 ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = int(epochs * 0.05)
        self.lr = lr
        self.grad_acc_step = grad_acc_step
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_schedular = lr_schedular
        self.device = device
        self.use_wandb = use_wandb
        self.runtime_path = runtime_path
        self.model_weight_name = "{}_ckpt.pt".format(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d-%H_%M_%S'))
        self.model_weight_path = self.runtime_path + self.model_weight_name
        self.log_path = runtime_path + "local.log"
        self.logger("New solver initialized...(:")
        self.logger("Optimizer: {}, Loss: {}, epochs:{}, lr:{}"
                    .format("Adam", self.criterion, self.epochs, self.lr))
        self.do_val = do_val
        self.save = save
        self.log_inter = log_inter

        self.load_pretrain = load_pretrain
        self.is_show_grad = True
        self.plot = True

        if self.use_wandb:
            wandb.init(project='transformer classifier',
                       name=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'))
            wandb.config = {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            }
            wandb.watch(self.model)

        # use 'attention is all you need' training strategy or not
        self.ailyn_strat = ailyn_strat

    def train(self, pretrained_model_path=""):
        grad_recorder = []
        weight_recorder = []
        self.model.train()

        if pretrained_model_path != "":
            self.load_model(pretrained_model_path)
            self.logger("Load pretrained model: {}".format(pretrained_model_path))

        self.model = self.model.to(self.device)
        mes = "Training on: {}".format(self.device)
        print(mes)
        self.logger(mes)
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            print("Training on epoch {}/{}".format(epoch, self.epochs))
            total_loss = 0
            num_iters = len(self.train_dataloader)
            grad_sum = torch.tensor(.0).to(self.device)
            for src, tgt in tqdm(self.train_dataloader):
                src = src.float().to(self.device)
                tgt = tgt.float().to(self.device)

                out = self.model(src)
                loss = self.criterion(out, tgt)
                total_loss += loss.item()
                loss.backward()

                if epoch % self.grad_acc_step == 0:
                    # grad probe
                    if self.is_show_grad:
                        grad_sum += self.check_grad()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            ws = torch.tensor(.0)
            for params in self.model.parameters():
                params = params.cpu().detach()
                ws += torch.pow(params, 2).sum()
            weight_recorder.append(ws)

            if epoch % self.grad_acc_step == 0 and self.lr_schedular is not None:
                self.lr_schedular.step()

            if self.ailyn_strat:
                self.adjust_lr(12, epoch)

            print("Epoch: {}, Loss = {:.3f}, Grad = {}, LR = {} "
                  .format(epoch, total_loss / num_iters,
                          grad_sum, self.optimizer.state_dict()['param_groups'][0]['lr']))
            grad_recorder.append(grad_sum)

            if self.use_wandb:
                wandb.log({'training loss per epoch': total_loss / num_iters})
                wandb.log({'learning rate per epoch': self.optimizer.state_dict()['param_groups'][0]['lr']})

            if (epoch - 1) % self.log_inter == 0:
                mes = "Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters)
                self.logger(mes)

            if self.do_val:
                print("Validating...")
                # val:
                res_str, val_loss = self.eval(self.val_dataloader)
                res_str = "Epoch {} Val loss: {:.3f}, Val Result: {}"\
                    .format(epoch, val_loss/len(self.val_dataloader), res_str)
                print(res_str)
                if (epoch - 1) % self.log_inter == 0:
                    self.logger(res_str)

            if epoch == 1 or epoch == epochs / 2:
                epoch_name = "{}_epoch_{}.pt".format(self.model_weight_name.split(".")[0], epoch)
                self.save_model(self.runtime_path + epoch_name)
                self.logger("Epoch {} weight saved at {}".format(epoch, epoch_name))

        duration = time.time() - start_time
        mes = "Training completed, time used: {} s".format(duration)
        print(mes)
        self.logger(mes)

        if self.save:
            self.save_model(self.model_weight_path)
            mes = "Model saved at: {}".format(self.model_weight_path)
            print(mes)
            self.logger(mes)

        if self.plot:
            y = []
            for item in grad_recorder:
                y.append(item.cpu())
            plt.plot(range(1, len(y)+1), y)
            plt.ylabel("Grad")
            plt.xlabel("Epoch")
            plt.title("gradient trend during training")
            time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d_%H_%M_%S')
            plt.savefig("train_grad_{}.png".format(time_str), dpi=200)
            plt.clf()
            y = []
            for item in weight_recorder:
                y.append(item.cpu())
            plt.plot(range(1, len(y)+1), y)
            plt.ylabel("Weight Norm")
            plt.xlabel("Epoch")
            plt.title("weight norm trend during training")
            time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d_%H_%M_%S')
            plt.savefig("train_weight_norm_{}.png".format(time_str), dpi=200)

    def test(self):
        assert self.test_dataloader is not None, "you need a test dataloader"
        assert self.model is not None, "no model loaded"
        print("Testing...")
        res_str, _ = self.eval(self.test_dataloader)
        res_str = "Test Result: " \
                  "{}".format(res_str)
        print(res_str)
        self.logger(res_str)

    def eval(self, dataloader):
        assert dataloader is not None
        self.model.eval()
        self.model = self.model.to(self.device)
        val_total = len(dataloader)
        num_neg, num_pos, num_cor, num_tps, num_prp, num_fns = 0, 0, 0, 0, 0, 0
        val_loss, acc, prc, rec, f1, tmp = .0, .0, 0, 0, 0, 0
        for src, tgt in tqdm(dataloader):
            gt = tgt[:, 1] >= 0.5
            num_gt_pos = int(gt.sum())
            len_gt = len(gt)

            num_pos += num_gt_pos
            num_neg += (len_gt - num_gt_pos)

            src = src.float().to(self.device)
            tgt = tgt.float().to(self.device)
            out = self.model(src)

            loss = self.criterion(out, tgt)
            val_loss += loss.item()

            pred = torch.argmax(out, dim=-1)
            pred = pred.cpu()
            tans = (pred == gt)
            num_cor += int(tans.sum())
            num_prp += int(pred.sum())
            num_tps += int((tans & gt).sum())
            num_fns += int(((0 == pred) & gt).sum())

        val_total = num_pos + num_neg
        if val_total != 0:
            acc = num_cor / val_total
        if num_prp != 0:
            prc = num_tps / num_prp
        if num_pos != 0:
            rec = num_tps / num_pos
        # or
        if num_fns + num_tps != 0:
            rec = num_tps / (num_fns + num_tps)
        if prc + rec != 0:
            f1 = 2 * prc * rec / (prc + rec)

        print("total={}".format(val_total))
        print("num_cor={}, num_tps={}, num_fns={}, num_prp={}, num_pos={}, num_neg={}"
              .format(num_cor, num_tps, num_fns, num_prp, num_pos, num_neg))
        res_str = "Acc: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(acc, prc, rec, f1)
        return res_str, val_loss

    def adjust_lr(self, dmodel, epoch):
        new_lr = dmodel ** (-0.5) * min(epoch ** (-0.5), epoch * self.warmup_epochs ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

    def logger(self, message):
        with open(self.log_path, "a") as f:
            time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
            f.write("[{}] {}\n".format(time_str, message))

    def check_grad(self):
        gs = torch.tensor(.0).to(self.device)
        for param in self.model.parameters():
            gs += torch.abs(param.grad).sum()
        return gs


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 500
    bs = 8
    lr = 1e-3
    disk_train_dataset = DiskDataset(train_csv_path, disk_data_path)
    disk_test_dataset = DiskDataset(test_csv_path, disk_data_path)
    disk_val_dataset = DiskDataset(val_csv_path, disk_data_path)
    grad_acc_step = 1

    transformer_classifier = TC(
        d_model=12,
        nhead=1,
        num_layers=3,
        dim_feedforward=64,
        hidden_size=32,
        # out_features=1,
    )

    # transformer_classifier = TransformerClr(
    #     d_model=12, nhead=3,
    #     num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, )

    lstm_classifier = LSTMPredictor(
        in_features=12,
        hidden_size=32,
        num_layers=6,
        out_features=2,
        device=device
    )

    Adam = torch.optim.Adam(transformer_classifier.parameters(), lr=lr, betas=(0.9, 0.98))

    CALR = torch.optim.lr_scheduler.CosineAnnealingLR(Adam, int(epochs))

    BCEL = nn.BCELoss()
    MSEL = nn.MSELoss()
    FCLL = FocalLoss()

    solver = Solver(
        device=device,
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        train_dataloader=DataLoader(disk_train_dataset, batch_size=bs, shuffle=False),
        test_dataloader=DataLoader(disk_test_dataset, batch_size=bs, shuffle=True),
        val_dataloader=DataLoader(disk_val_dataset, batch_size=bs, shuffle=True),
        model=transformer_classifier,
        optimizer=Adam,
        criterion=BCEL,
        lr_schedular=CALR,
        use_wandb=False,
        do_val=True,
        grad_acc_step=grad_acc_step,
        save=True,
        log_inter=int(epochs/10),
        load_pretrain=False,
        ailyn_strat=False
    )

    # solver.train()
    solver.train(pretrained_model_path="./run/init/2023-05-15-12_53_03_maml.pt")
    # solver.load_model(solver.model_weight_path)
    # solver.load_model("run/2023-05-11-17_49_18_ckpt_epoch_1.pt")
    solver.test()
