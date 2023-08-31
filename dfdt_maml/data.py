import os
import numpy as np
import pandas as pd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torch import nn, tensor, Tensor
from tqdm import tqdm
from utils import check_dir

LOG_LENGTH = 30
LENGTH_LIM = 20


def make_train_test_val(csv_path, target_path=".", pn_factor=4, test_factor=0.3, val_factor=0.1):

    df = pd.read_csv(csv_path)
    total = len(df)
    counts = df["fault"].value_counts()
    num_failure = counts[1]
    factor_failure = num_failure / total
    num_test = total * test_factor
    num_val = total * val_factor
    num_train = total - num_test - num_val

    num_train_f = int(num_train * factor_failure)
    num_train_expected = num_train_f * pn_factor
    num_train_g = min(num_train - num_train_expected, num_train_expected)

    num_test_f = int(num_test * factor_failure)
    num_test_expected = num_test_f * pn_factor
    num_test_g = min(num_test - num_test_expected, num_test_expected)

    num_val_f = int(num_val * factor_failure)
    num_val_expected = num_val_f * pn_factor
    num_val_g = min(num_val - num_val_expected, num_val_expected)

    print("Train samples: {} good disks and {} failures, total: {}"
          .format(num_train_g, num_train_f, num_train_g + num_train_f))
    print("Test samples: {} good disks and {} failures, total: {}"
          .format(num_test_g, num_test_f, num_test_g + num_test_f))
    print("Validation samples: {} good disks and {} failures, total: {}"
          .format(num_val_g, num_val_f, num_val_g + num_val_f))

    good_df = df[df["fault"] == 0]
    failure_df = df[df["fault"] == 1]

    trainval_df_g = good_df.sample(num_train_g + num_val_g)
    test_df_g = good_df[~good_df.index.isin(trainval_df_g.index)].sample(num_test_g)
    train_df_g = trainval_df_g.sample(num_train_g)
    val_df_g = trainval_df_g[~trainval_df_g.index.isin(train_df_g.index)]

    print("trian good: ", len(train_df_g))
    print("test good: ", len(test_df_g))
    print("val good: ", len(val_df_g))

    trainval_df_f = failure_df.sample(num_train_f + num_val_f)
    test_df_f = failure_df[~failure_df.index.isin(trainval_df_f.index)].sample(num_test_f)
    train_df_f = trainval_df_f.sample(num_train_f)
    val_df_f = trainval_df_f[~trainval_df_f.index.isin(train_df_f.index)]

    print("train bad: ", len(train_df_f))
    print("test bad: ", len(test_df_f))
    print("val bad: ", len(val_df_f))

    train_df = pd.concat([train_df_g, train_df_f])
    test_df = pd.concat([test_df_g, test_df_f])
    val_df = pd.concat([val_df_g, val_df_f])

    print("Train sample: ", len(train_df))
    print("Val sample: ", len(val_df))
    print("Test sample: ", len(test_df))

    train_df.to_csv(os.path.join(target_path, "train_sample.csv"), index=False)
    test_df.to_csv(os.path.join(target_path, "test_sample.csv"), index=False)
    val_df.to_csv(os.path.join(target_path, "val_sample.csv"), index=False)


def drop_too_few():
    stages = ["train", "val", "test"]
    for stage in stages:
        csv_path = "data/{}_sample.csv".format(stage)
        removed = drop_too_few_one(csv_path, "data/preprocessed")
        print("Remove {} disks in {} sample that logs under {}".format(removed, stage, LENGTH_LIM))


def drop_too_few_one(csv_path, data_path):
    ds = DiskDataset(csv_path, data_path, trial=True)
    blacklist = []
    for t, l, _ in ds:
        if len(t) <= LENGTH_LIM:
            blacklist.append(_)
    data_df = pd.read_csv(csv_path)
    len_before = len(data_df)
    data_df = data_df[~data_df["serial_number"].isin(blacklist)]
    data_df.to_csv(csv_path, index=False)
    return len_before-len(data_df)


def make_episodic(csv_path, target_path="./episodic", num_tasks=100, num_k=100):
    df = pd.read_csv(csv_path)
    good_df = df[df["fault"] == 0]
    failure_df = df[df["fault"] == 1]
    train_path = os.path.join(target_path, "meta_train/")
    test_path = os.path.join(target_path, "meta_test/")
    check_dir(target_path)
    check_dir(train_path)
    check_dir(test_path)
    for task in range(num_tasks):
        task_train_path = os.path.join(train_path, "task_{}_train.csv".format(task))
        task_test_path = os.path.join(test_path, "task_{}_test.csv".format(task))
        train_df_g = good_df.sample(int(num_k/2))
        test_df_g = good_df.sample(int(num_k/2))
        train_df_f = failure_df.sample(int(num_k/2))
        test_df_f = failure_df.sample(int(num_k/2))
        train_df = pd.concat([train_df_g, train_df_f])
        test_df = pd.concat([test_df_g, test_df_f])
        train_df.to_csv(task_train_path, index=False)
        test_df.to_csv(task_test_path, index=False)
        r1 = drop_too_few_one(task_train_path, "data/preprocessed")
        r2 = drop_too_few_one(task_test_path, "data/preprocessed")
        print("Task {} dataset, trains: {}, test: {}".format(task, num_k-r1, num_k-r2))


def bisearch(df, dt) -> int:
    arr = df.to_numpy(dtype=np.int32)
    dt = int(dt)
    lf, rt = 0, len(arr)
    mid = 0
    while lf < rt:
        mid = int((lf + rt) / 2)
        if arr[mid] == dt:
            break
        elif arr[mid] < dt:
            lf = mid+1
        else:
            rt = mid
    return mid


class DiskDataset(Dataset):
    def __init__(self, csv_path: str, data_path: str, trial=False):
        super(DiskDataset, self).__init__()
        self.csv_path = csv_path
        self.data_path = data_path
        self.csv_file = open(csv_path, "r")
        self.csv_lines = self.csv_file.readlines()[1:]
        self.num_lines = len(self.csv_lines) - 1
        self.trial = trial

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        entry = self.csv_lines[index].strip().split(",")
        serial, failure = entry[0], int(entry[1])
        disk_df = pd.read_csv(os.path.join(self.data_path, "{}.csv".format(serial)))

        if failure == 1:
            cut_index = bisearch(disk_df["dt"], entry[2])
        else:
            cut_index = len(disk_df)
        disk_df = disk_df[max(0, cut_index-LOG_LENGTH):cut_index]

        disk_df = disk_df.drop(columns=["dt"])
        disk_np = np.array(disk_df)

        if len(disk_np) < 30:
            pad = np.zeros((30-len(disk_np), 12))
            disk_np = np.concatenate([pad, disk_np])

        disk_ts = tensor(disk_np) # N * 12
        # disk_ts = tensor(disk_np)
        disk_ts = torch.nn.functional.normalize(disk_ts, dim=1)

        # reg:
        if failure == 0:
            tgt = tensor([1, 0])
        else:
            tgt = tensor([0, 1])
        # cls:
        # tgt = tensor([failure])
        if self.trial:
            return disk_ts, tgt, serial
        else:
            return disk_ts, tgt


if __name__ == "__main__":
    # make_episodic("data/disk_info.csv", target_path="data/episodic")
    # make_train_test_val("data/disk_info.csv", target_path="data/", pn_factor=1)

    # drop_too_few()

    dd = DiskDataset("./data/train_sample.csv", "./data/preprocessed/")
    target, label = dd[29]
    cnt = 0
    for i, t in dd:
        if len(i) != 30:
            print("Fuck {}".format(len(i)))

    print(len(dd))
    print(target.shape)
    print(label)
