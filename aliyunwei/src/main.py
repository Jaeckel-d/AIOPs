# 序列格式微调  加入位置position  
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.optim.lr_scheduler import *
from model.utils import macro_f1, FGM
from model.dataset import MyDataSet
# from model.model import MyModel
from model.model_v2 import MyModel_V2
import random
import os
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

train_set = pd.read_csv("../tmp_data/train_set.csv")
test_seta = pd.read_csv("../tmp_data/test_set_a.csv")
test_seta_label = pd.read_csv("../data/test_ab/preliminary_test_label_dataset_a.csv")
test_seta = test_seta.merge(test_seta_label, on=['sn', 'fault_time'], how='inner')
test_setb = pd.read_csv("../tmp_data/test_set_b.csv")
test_setb_label = pd.read_csv("../data/test_ab/preliminary_test_label_dataset_b.csv")
test_setb = test_setb.merge(test_setb_label, on=['sn', 'fault_time'], how='inner')

# train_set['positive_p'] = 1
# train_set = pd.concat([train_set, test_seta, test_setb])

# finala_set = pd.read_csv("../tmp_data/finala_set.csv")
finala_set = pd.read_csv('../tmp_data/test_set_a.csv')


def train_and_evaluate(train_set_, test_set_, submit_set_, name, att_cate='gate'):
    model = MyModel_V2(att_cate=att_cate)
    fgm = FGM(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    # lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet(test_set_, mode='test')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    train_data = iter(train_set_)
    test_data = iter(test_set_)
    best_f1 = 0
    for epoch in range(35):
        running_loss = 0
        for step in range(train_set_.step_max):
            feat, label = next(train_data)
            model.train()
            pred = model(feat)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            fgm.attack()
            loss_adv = criterion(model(feat), label)
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            running_loss += loss.item()
            if step % 50 == 49:
                print(f"Epoch {epoch + 1}, step {step + 1}: {running_loss}")
                running_loss = 0
                # lr_sche.step()
        # epoch 结束后 evaluate
        preds = []
        with torch.no_grad():
            for step in range(test_set_.step_max):
                model.eval()
                feat, label = next(test_data)
                pred = model(feat).argmax(dim=-1).numpy()
                preds.extend(pred)
        test_df[f'pred_{att_cate}_{name}'] = preds
        macro_F1 = macro_f1(test_df, f'pred_{att_cate}_{name}')
        if macro_F1 > best_f1:
            best_f1 = macro_F1
            torch.save(model, f'../src/{att_cate}_{name}.pt')
            try:
                test_df.to_csv(f"../submission/pred_{att_cate}.csv", index=False)
            except FileNotFoundError:
                pass
        print(f"Macro F1: {macro_F1}")
        print(f"Max Macro F1: {best_f1}")
        scheduler.step(macro_F1)
    print('Max macro F1:', best_f1)
    # submit_set = MyDataSet(submit_set_, mode='predict')
    # submit_set_iter = iter(submit_set)
    # preds = []
    # model.load_state_dict(torch.load(f'../model/model_{att_cate}_{name}.pt'))
    # with torch.no_grad():
    #     model.eval()
    #     for step in range(submit_set.step_max):
    #         feat = next(submit_set_iter)
    #         pred = model(feat)
    #         pred = torch.softmax(pred, dim=-1).numpy()
    #         pred = [json.dumps(p.tolist()) for p  in pred]
    #         preds.extend(pred)
    # submit_set_[f'label_{att_cate}_{name}'] = preds
    # submit_set_.to_csv(f'../submission/submit.csv', index=False)


# for i, (train_idx, test_idx) in enumerate(
#         StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(train_set, train_set.label)):
#     train_set_ = train_set.iloc[train_idx]
#     train_set_ = pd.concat([train_set_, train_set_[train_set_.label == 0]]).reset_index(drop=True)
#     test_set_ = train_set.iloc[test_idx]
#     train_and_evaluate(train_set_, test_set_, finala_set, i, att_cate='pool')
#     print('=====================================')


def get_feature(dataset, name, att_cate='gate'):
    model = MyModel_V2(att_cate=att_cate, return_feature=10)
    model_dict = torch.load(f"{att_cate}_0.pt").state_dict()
    model.load_state_dict(model_dict)
    test_data = MyDataSet(dataset, mode="test", batch_size=1)
    test_data_iter = iter(test_data)
    feature_less, feature_more, labels = [], [], []
    with torch.no_grad():
        model.eval()
        for step in range(test_data.step_max):
            feat, label = next(test_data_iter)
            feat_less, feat_more = model(feat)
            feature_less.append(feat_less.numpy())
            feature_more.append(feat_more.numpy())
            labels.append(label.numpy())
    data_dict = {"label": labels, "feature_less": feature_less, "feature_more": feature_more}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{att_cate}_feature.csv")


if __name__ == "__main__":
    get_feature(test_setb, name='test', att_cate='pool')
# print("=========================  模型1训练结束  ===========================")

# for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=2022).split(train_set, train_set.label)):
#     train_set_ = train_set.iloc[train_idx]
#     train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
#     test_set_ = train_set.iloc[test_idx]     
#     train_and_evaluate(train_set_, test_set_, finala_set, i, att_cate='lstm1')
#     print('=====================================')

# print("=========================  模型2训练结束  ===========================")
