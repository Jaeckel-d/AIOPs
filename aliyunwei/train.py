import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AttentionPool, AttentionPool_
from aiop_dataset import AIOPSTest, AIOPSVal, AIOPS
import torch.optim as optim
from tqdm import tqdm
from loss import BCEFocalLoss
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from utils import move_to_device
import datetime as dt
import wandb
import time
import argparse
from utils import print_args, count_label
from torchsampler import ImbalancedDatasetSampler
from helper import one_hot_embedding
from loss import relu_evidence, edl_digamma_loss

seed = 42
torch.manual_seed(seed)


class Solver:
    def __init__(self,
                 batch_size=64, epochs=100, lr=0.001,
                 train_dataloader=None, test_dataloader=None, val_dataloader=None,
                 model=None, optimizer=None, criterion=None, lr_schedular=None, grad_acc_step=3,
                 use_wandb=False, runtime_path="./run/", do_val=False,
                 save=False, log_inter=50, device=None, flag=False, binary=False, num_classes=3
                 ):
        self.batch_size = batch_size
        self.epochs = epochs
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
        self.log_path = os.path.join(runtime_path, "local.log")
        self.logger("New solver initialized...(:")
        self.logger("Optimizer: {}, Loss: {}, epochs:{}, lr:{}"
                    .format("Adam", self.criterion, self.epochs, self.lr))
        self.do_val = do_val
        self.save = save
        self.log_inter = log_inter
        self.flag = flag
        self.binary = binary
        self.num_classes = num_classes

        if self.use_wandb:
            wandb.init(project='AttentionPool classifier',
                       name=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'))
            wandb.config = {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            }
            wandb.watch(self.model)

    def train(self):
        self.model.train()
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
            for src, tgt in tqdm(self.train_dataloader):
                # src = src.float().to(self.device)
                src = move_to_device(src, self.device)
                tgt = torch.unsqueeze(tgt, dim=1)
                tgt = tgt.float().to(self.device)

                # tgt = tgt.float().to(self.device)
                out = self.model(src)
                loss = self.criterion(out, tgt)
                total_loss += loss.item()
                loss.backward()
                if (epoch + 1) % self.grad_acc_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if (epoch + 1) % self.grad_acc_step == 0 and self.lr_schedular is not None:
                self.lr_schedular.step()

            print("Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters))
            if self.use_wandb:
                wandb.log({'training loss per epoch': total_loss / num_iters})
                wandb.log({'learning rate per epoch': self.optimizer.state_dict()['param_groups'][0]['lr']})
            if (epoch - 1) % self.log_inter == 0:
                mes = "Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters)
                self.logger(mes)

            if self.do_val:
                if (epoch - 1) % self.log_inter == 0:
                    print("Validating...")
                    # val:
                    result, pre_base, pre_nov, rec_base, rec_nov, f1_base, f1_nov, acc = self.eval(self.val_dataloader)
                    self.logger(result)

        duration = time.time() - start_time
        mes = "Training completed, time used: {} s".format(duration)
        print(mes)
        self.logger(mes)

        if self.save:
            self.save_model(self.model_weight_path)
            mes = "Model saved at: {}".format(self.model_weight_path)
            print(mes)
            self.logger(mes)

    def train_evidence(self):
        self.model.train()
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
            for src, tgt in tqdm(self.train_dataloader):
                # src = src.float().to(self.device)
                src = move_to_device(src, self.device)
                y = one_hot_embedding(labels=tgt, num_classes=self.num_classes)
                y = y.to(self.device)
                out = self.model(src)
                _, pred = torch.max(out, 1)
                loss = self.criterion(out, y.float(), epoch, self.num_classes, 10, self.device)

                tgt = tgt.float().to(self.device)
                match = torch.reshape(torch.eq(pred, tgt).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(out)
                alpha = evidence + 1
                u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                total_loss += loss.item()
                loss.backward()
                if (epoch + 1) % self.grad_acc_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if (epoch + 1) % self.grad_acc_step == 0 and self.lr_schedular is not None:
                self.lr_schedular.step()

            print("Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters))
            if self.use_wandb:
                wandb.log({'training loss per epoch': total_loss / num_iters})
                wandb.log({'learning rate per epoch': self.optimizer.state_dict()['param_groups'][0]['lr']})
            if (epoch - 1) % self.log_inter == 0:
                mes = "Epoch: {}, Loss = {:.3f}".format(epoch, total_loss / num_iters)
                self.logger(mes)

            if self.do_val:
                if (epoch - 1) % self.log_inter == 0:
                    print("Validating...")
                    # val:
                    self.eval_evidence(self.val_dataloader)
                    # self.logger(result)

        duration = time.time() - start_time
        mes = "Training completed, time used: {} s".format(duration)
        print(mes)
        self.logger(mes)

        if self.save:
            self.save_model(self.model_weight_path)
            mes = "Model saved at: {}".format(self.model_weight_path)
            print(mes)
            self.logger(mes)

    def test(self):
        assert self.test_dataloader is not None, "you need a test dataloader"
        assert self.model is not None, "no model loaded"
        print("Testing...")
        res_str, pre_base, pre_nov, rec_base, rec_nov, f1_base, f1_nov, acc = self.eval(self.test_dataloader)
        res_str = "Test Result: " \
                  "{}".format(res_str)
        print(res_str)
        self.logger(res_str)
        return pre_base, pre_nov, rec_base, rec_nov, f1_base, f1_nov, acc

    def eval(self, dataloader):
        assert dataloader is not None
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            y_pred = []
            y_label = []
            for src, tgt in tqdm(dataloader):
                src = move_to_device(src, self.device)
                label = tgt.to(self.device)
                out = self.model(src)
                prediction = (out >= 0.5).float()

                y_pred.append(int(prediction))
                y_label.append(int(label))
            results = classification_report(y_true=y_label, y_pred=y_pred)
            precision_0 = precision_score(y_label, y_pred, pos_label=0)
            precision_1 = precision_score(y_label, y_pred, pos_label=1)
            recall_0 = recall_score(y_label, y_pred, pos_label=0)
            recall_1 = recall_score(y_label, y_pred, pos_label=1)
            f1_0 = f1_score(y_label, y_pred, pos_label=0)
            f1_1 = f1_score(y_label, y_pred, pos_label=1)
            acc = accuracy_score(y_label, y_pred)
            print(results)
            self.logger(results)

        if self.flag:
            return results, precision_1, precision_0, recall_1, recall_0, f1_1, f1_0, acc
        else:
            return results, precision_0, precision_1, recall_0, recall_1, f1_0, f1_1, acc

    def eval_evidence(self, dataloader):
        assert dataloader is not None
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            y_pred = []
            y_label = []
            for src, tgt in tqdm(dataloader):
                src = move_to_device(src, self.device)
                label = tgt.to(self.device)

                output = self.model(src)
                evidence = relu_evidence(output)
                alpha = evidence + 1
                uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(output, 1)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                output = output.flatten()
                prob = prob.flatten()
                preds = preds.flatten()
                # print("Predict:", preds[0])
                # print("Probs:", prob)
                # print("Uncertainty:", uncertainty)
                print("Label:", label.item(), " Predict:", preds.item(), " Probability:", prob.tolist(),
                      " Uncertainty:", uncertainty.item())
                # prediction = (out >= 0.5).float()

                # todo:此处逻辑有待考证
                if uncertainty > 0.5:
                    preds = torch.tensor(self.num_classes)

                y_pred.append(int(preds))
                y_label.append(int(label))
            results = classification_report(y_true=y_label, y_pred=y_pred)
            # precision_0 = precision_score(y_label, y_pred, pos_label=0)
            # precision_1 = precision_score(y_label, y_pred, pos_label=1)
            # recall_0 = recall_score(y_label, y_pred, pos_label=0)
            # recall_1 = recall_score(y_label, y_pred, pos_label=1)
            # f1_0 = f1_score(y_label, y_pred, pos_label=0)
            # f1_1 = f1_score(y_label, y_pred, pos_label=1)
            # acc = accuracy_score(y_label, y_pred)
            print(results)
            self.logger(results)

        # if self.flag:
        #     return results, precision_1, precision_0, recall_1, recall_0, f1_1, f1_0, acc
        # else:
        #     return results, precision_0, precision_1, recall_0, recall_1, f1_0, f1_1, acc

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

    def logger(self, message):
        with open(self.log_path, "a") as f:
            time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
            f.write("[{}] {}\n".format(time_str, message))


def train_once(epochs=100, bs=4, lr=1e-3, base=None, novel=None, k_shot=1, focal=False, imb_sampler=False, alpha=0.25):
    if novel is None:
        novel = [2, ]
    if base is None:
        base = [1, ]

    flag = False
    if base[0] > novel[0]:
        flag = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    train_set = AIOPS(root_path="./aliyunwei/tmp_data", split="train", k_shot=k_shot, base=base, novel=novel)
    test_set = AIOPSTest(root_path="./aliyunwei/tmp_data", split="test", base=base, novel=novel)
    val_set = AIOPSVal(root_path="./aliyunwei/tmp_data", split='val', base=base, novel=novel)

    if imb_sampler:
        train_data = DataLoader(dataset=train_set, batch_size=bs, num_workers=4,
                                sampler=ImbalancedDatasetSampler(train_set))
        count_label(train_data)
    else:
        train_data = DataLoader(dataset=train_set, batch_size=bs, num_workers=4, shuffle=True)
        count_label(train_data)
    val_data = DataLoader(dataset=val_set, batch_size=1, num_workers=4, shuffle=False)
    test_data = DataLoader(dataset=test_set, batch_size=1, num_workers=4, shuffle=False)

    model = AttentionPool_()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    BCEL = nn.BCEWithLogitsLoss()
    BCEFOCAL = BCEFocalLoss(alpha=alpha)

    # criterion = BCEFocalLoss()
    if focal:
        criterion = BCEFOCAL
    else:
        criterion = BCEL

    criterion = edl_digamma_loss

    solver = Solver(
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        device=device,
        train_dataloader=train_data,
        test_dataloader=test_data,
        val_dataloader=val_data,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_schedular=schedular,
        grad_acc_step=1,
        do_val=True,
        save=True,
        log_inter=int(epochs / 10),
        runtime_path="./run/",
        # use_wandb=True
        flag=flag,
    )

    # solver.train()
    solver.train_evidence()
    return solver.test()


def train_evidence_once(epochs=10, bs=4, lr=1e-3, train_class=None, test_class=None, imb_sampler=None):
    if train_class is None:
        train_class = [0, 1, 2]
    if test_class is None:
        test_class = [0, 1, 2, 3]

    # flag = False
    # if base[0] > novel[0]:
    #     flag = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    train_set = AIOPSTest(root_path="./aliyunwei/tmp_data", split="train", base=train_class, novel=None)
    test_set = AIOPSTest(root_path="./aliyunwei/tmp_data", split="test", base=test_class, novel=None)
    val_set = AIOPSVal(root_path="./aliyunwei/tmp_data", split='val', base=test_class, novel=None)

    if imb_sampler:
        train_data = DataLoader(dataset=train_set, batch_size=bs, num_workers=4,
                                sampler=ImbalancedDatasetSampler(train_set))
        count_label(train_data)
    else:
        train_data = DataLoader(dataset=train_set, batch_size=bs, num_workers=4, shuffle=True)
        count_label(train_data)
    val_data = DataLoader(dataset=val_set, batch_size=1, num_workers=4, shuffle=False)
    test_data = DataLoader(dataset=test_set, batch_size=1, num_workers=4, shuffle=False)

    model = AttentionPool_()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    # BCEL = nn.BCEWithLogitsLoss()
    # BCEFOCAL = BCEFocalLoss(alpha=alpha)

    # criterion = BCEFocalLoss()
    # if focal:
    #     criterion = BCEFOCAL
    # else:
    #     criterion = BCEL

    criterion = edl_digamma_loss

    solver = Solver(
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        device=device,
        train_dataloader=train_data,
        test_dataloader=test_data,
        val_dataloader=val_data,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_schedular=schedular,
        grad_acc_step=1,
        do_val=True,
        save=True,
        log_inter=int(epochs / 10),
        runtime_path="./run/",
        # use_wandb=True
        # flag=flag,
    )

    # solver.train()
    solver.train_evidence()
    return solver.eval_evidence(solver.test_dataloader)


def maml_test(base=None, novel=None, model_path=""):
    if novel is None:
        novel = [3, ]
    if base is None:
        base = [0, ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    test_set = AIOPSTest(root_path="./aliyunwei/tmp_data", split="test", base=base, novel=novel)
    test_data = DataLoader(dataset=test_set, batch_size=1, num_workers=4, shuffle=False)

    model = AttentionPool_()
    solver = Solver(
        device=device,
        test_dataloader=test_data,
        model=model,
    )
    solver.load_model(model_path)
    return solver.test()


def baseline_train_test():
    repeat_num = 100
    args, name, result = init()
    for i in range(repeat_num):
        x, y, z, a, b, c, acc = train_once(epochs=args.epoch, bs=args.bs, lr=args.lr, base=args.base, novel=args.novel,
                                           k_shot=args.k_shot,
                                           focal=args.focal, imb_sampler=args.imb, alpha=args.alpha)
        ans = [x, y, z, a, b, c, acc]
        for j, l in enumerate(ans):
            result[j].append(l)

    result_path = "lr_" + str(args.lr) + "_base_" + str(args.base) + "_novel_" + str(args.novel) + "_k_" + str(
        args.k_shot) + "_focal_" + str(args.focal) + "_sample_" + str(args.imb) + ".txt"

    if not os.path.exists("./" + args.exp):
        os.mkdir("./" + args.exp)

    with open(os.path.join("./" + args.exp, result_path), 'w') as f:
        name__ = "\t".join(name)
        f.write(name__ + "\n")
        for i in range(len(result[0])):
            f.write(str(result[0][i]) + "\t" + str(result[1][i]) + "\t" + str(result[2][i]) + "\t" +
                    str(result[3][i]) + "\t" + str(result[4][i]) + "\t" + str(result[5][i]) + "\t" +
                    str(result[6][i]) + "\n"
                    )
        f.write("--------------------------mean-----------------------+\n")
        f.write(str(sum(result[0]) / len(result[0])) + "\t" +
                str(sum(result[1]) / len(result[1])) + "\t" +
                str(sum(result[2]) / len(result[2])) + "\t" +
                str(sum(result[3]) / len(result[3])) + "\t" +
                str(sum(result[4]) / len(result[4])) + "\t" +
                str(sum(result[5]) / len(result[5])) + "\t" +
                str(sum(result[6]) / len(result[6])) + "\n"
                )
        # print(precision_base)
        # print(precision_novel)
        # print(recall_base)
        # print(recall_novel)
        # print(f1_base)
        # print(f1_novel)
        #
        # print(sum(precision_base)/len(precision_base))
        # print(sum(precision_novel)/len(precision_novel))
        # print(sum(recall_base)/len(recall_base))
        # print(sum(recall_novel)/len(recall_novel))
        # print(sum(f1_base)/len(f1_base))
        # print(sum(f1_novel)/len(f1_novel))


def init():
    parser = argparse.ArgumentParser(description='imbalance data.')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--bs', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--base', nargs="+", type=int, default=[0, ], help='base class')
    parser.add_argument('--novel', nargs="+", type=int, default=[1, ], help='novel class')
    parser.add_argument('--k_shot', type=int, default=1, help="k-shot")
    parser.add_argument('--focal', type=bool, default=True, help="use focal loss")
    parser.add_argument('--alpha', type=float, default=0.75, help="focal loss alpha")
    parser.add_argument('--imb', type=bool, default=True, help="imb_sampler")
    parser.add_argument('--exp', type=str, default="/100/baseline_imb_focal_0.75", help="exp name")
    args_ = parser.parse_args()  # 获取所有参数
    name_ = ["precision_base", "precision_novel", "recall_base", "recall_novel", "f1_base", "f1_novel", "acc"]
    result_ = [[], [], [], [], [], [], []]
    print_args(parser, args_)
    return args_, name_, result_


def maml_test_init():
    parser = argparse.ArgumentParser(description='maml test.')
    parser.add_argument('--k_shot', type=int, default=1, help="k-shot")
    parser.add_argument('--base', nargs="+", type=int, default=[0, ], help='base class')
    parser.add_argument('--novel', nargs="+", type=int, default=[3, ], help='novel class')
    parser.add_argument('--model_path', type=str,
                        default="/home/wyd/AIOP/MAML/meta-finetune/MAML_1_shot_finetuneclass_0",
                        help="model_path")
    parser.add_argument('--exp', type=str, default="FOMAML", help="exp name")
    args_ = parser.parse_args()  # 获取所有参数
    name_ = ["precision_base", "precision_novel", "recall_base", "recall_novel", "f1_base", "f1_novel", "acc"]
    result_ = [[], [], [], [], [], [], []]
    print_args(parser, args_)
    return args_, name_, result_


def meta_test():
    args, name, result = maml_test_init()

    weights = os.listdir(args.model_path)
    for w in weights:
        x, y, z, a, b, c, acc = maml_test(base=args.base, novel=args.novel, model_path=os.path.join(args.model_path, w))
        ans = [x, y, z, a, b, c, acc]
        for j, l in enumerate(ans):
            result[j].append(l)

    result_path = args.exp + "_base_" + str(args.base) + "_novel_" + str(args.novel) + "_k_" + str(
        args.k_shot) + ".txt"

    if not os.path.exists("./100/" + args.exp):
        os.mkdir("./100/" + args.exp)

    with open(os.path.join("./100/" + args.exp, result_path), 'w') as f:
        name__ = "\t".join(name)
        f.write(name__ + "\n")
        for i in range(len(result[0])):
            f.write(str(result[0][i]) + "\t" + str(result[1][i]) + "\t" + str(result[2][i]) + "\t" +
                    str(result[3][i]) + "\t" + str(result[4][i]) + "\t" + str(result[5][i]) + "\t" +
                    str(result[6][i]) + "\n"
                    )
        f.write("--------------------------mean-----------------------+\n")
        f.write(str(sum(result[0]) / len(result[0])) + "\t" +
                str(sum(result[1]) / len(result[1])) + "\t" +
                str(sum(result[2]) / len(result[2])) + "\t" +
                str(sum(result[3]) / len(result[3])) + "\t" +
                str(sum(result[4]) / len(result[4])) + "\t" +
                str(sum(result[5]) / len(result[5])) + "\t" +
                str(sum(result[6]) / len(result[6])) + "\n"
                )
        # print(precision_base)
        # print(precision_novel)
        # print(recall_base)
        # print(recall_novel)
        # print(f1_base)
        # print(f1_novel)
        #
        # print(sum(precision_base)/len(precision_base))
        # print(sum(precision_novel)/len(precision_novel))
        # print(sum(recall_base)/len(recall_base))
        # print(sum(recall_novel)/len(recall_novel))
        # print(sum(f1_base)/len(f1_base))
        # print(sum(f1_novel)/len(f1_novel))


if __name__ == '__main__':
    # baseline_train_test()
    # meta_test()
    train_evidence_once()
