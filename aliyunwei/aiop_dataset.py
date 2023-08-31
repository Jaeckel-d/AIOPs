import os
import random

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import json


class AIOPS(Dataset):
    def __init__(self, root_path, split="train", k_shot=1, base=None, novel=None, normalize=False,
                 transform=None):
        super(AIOPS, self).__init__()

        if base is None:
            base = [0, ]
        if novel is None:
            novel = [1, ]

        split_dict = {'train': 'train_set',  # standard train
                      'val': 'a_test_set',
                      'test': 'b_test_set',
                      'meta-train': 'train_set',  # meta-train
                      'meta-val': 'a_test_set',  # meta-val
                      'meta-test': 'b_test_set',  # meta-test
                      }

        split_tag = split_dict[split]

        split_file_path = os.path.join(root_path, split_tag + '.csv')
        assert os.path.exists(split_file_path)
        split_file = pd.read_csv(split_file_path)
        # 1476 3380 9939 2457
        split_file_base = split_file.loc[split_file['label'].isin(base)]
        split_file_novel = [split_file.loc[split_file['label'].isin([nov])].sample(k_shot) for nov in novel]
        split_file_novel.append(split_file_base)
        split_file = pd.concat(split_file_novel, axis=0, ignore_index=True)

        seed = 42
        split_file = split_file.sample(frac=1, random_state=seed)

        data = split_file.drop("label", axis=1)
        label = split_file['label']
        # label = split_file.loc[split_file['label'].isin([1,2])]

        msg_all, msg_mask = [], []
        venus_all, venus_mask = [], []
        msg_sentence_max, venus_sentence_max = 0, 0

        for sample in list(data.iloc[:].msg_feature.values):
            msgs = json.loads(sample)
            msg_all.append(torch.tensor(msgs))
            msg_mask.append(torch.ones(len(msgs)))
            msg_sentence_max = max(msg_sentence_max, len(msgs))

        for sample in list(data.iloc[:].venus_feature.values):
            venus = json.loads(sample)
            venus_all.append(torch.tensor(venus))
            venus_mask.append(torch.ones(len(venus)))
            venus_sentence_max = max(venus_sentence_max, len(venus))

        msg_all = np.array([F.pad(sample, (0, 0, 0, msg_sentence_max - sample.shape[0])).numpy() for sample in msg_all])
        msg_all = torch.tensor(msg_all)
        venus_all = pad_sequence(venus_all, batch_first=True)
        msg_mask = pad_sequence(msg_mask, batch_first=True)
        venus_mask = pad_sequence(venus_mask, batch_first=True)
        server_model = torch.tensor(list(data.iloc[:].server_model))
        crash_dump = torch.tensor([json.loads(line) for line in list(data.iloc[:].crashdump_feature)],
                                  dtype=torch.int)

        data = []
        # self.data = torch.stack([msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump], dim=0)
        for q, w, e, r, t, y in zip(msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump):
            data.append([q, w, e, r, t, y])
        self.data = data

        label = np.array(label)
        label_key = sorted(np.unique(label))

        encode_labels = np.arange(len(label_key))
        label_map = dict(zip(label_key, encode_labels))
        label = np.array([label_map[lab] for lab in label])

        self.root_path = root_path
        self.split_tag = split_tag
        self.label = label
        self.n_classes = len(label_key)

        self.base = base
        self.novel = novel
        self.k_shot = k_shot

        if normalize:
            pass

        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def get_labels(self):
        return self.label


class AIOPSTest(AIOPS):
    def __init__(self, root_path, split="test", base=None, novel=None, normalize=False,
                 transform=None):
        super(AIOPS, self).__init__()

        if base is None:
            base = [0, ]
        if novel is None:
            novel = [1, ]

        split_dict = {'train': 'train_set',  # standard train
                      'val': 'a_test_set',
                      'test': 'b_test_set',
                      'meta-train': 'train_set',  # meta-train
                      'meta-val': 'a_test_set',  # meta-val
                      'meta-test': 'b_test_set',  # meta-test
                      }

        split_tag = split_dict[split]

        split_file_path = os.path.join(root_path, split_tag + '.csv')
        assert os.path.exists(split_file_path)
        split_file = pd.read_csv(split_file_path)

        split_file = split_file.loc[split_file['label'].isin(base + novel)]
        counts = split_file["label"].value_counts()
        min_count = counts.min()
        result = pd.DataFrame()
        for name, group in split_file.groupby('label'):
            result = pd.concat([result, group.nsmallest(n=min_count, columns="label")])
        split_file = result

        seed = 42
        split_file = split_file.sample(frac=1, random_state=seed)

        data = split_file.drop("label", axis=1)
        label = split_file['label']

        msg_all, msg_mask = [], []
        venus_all, venus_mask = [], []
        msg_sentence_max, venus_sentence_max = 0, 0

        for sample in list(data.iloc[:].msg_feature.values):
            msgs = json.loads(sample)
            msg_all.append(torch.tensor(msgs))
            msg_mask.append(torch.ones(len(msgs)))
            msg_sentence_max = max(msg_sentence_max, len(msgs))

        for sample in list(data.iloc[:].venus_feature.values):
            venus = json.loads(sample)
            venus_all.append(torch.tensor(venus))
            venus_mask.append(torch.ones(len(venus)))
            venus_sentence_max = max(venus_sentence_max, len(venus))

        msg_all = np.array([F.pad(sample, (0, 0, 0, msg_sentence_max - sample.shape[0])).numpy() for sample in msg_all])
        msg_all = torch.tensor(msg_all)
        venus_all = pad_sequence(venus_all, batch_first=True)
        msg_mask = pad_sequence(msg_mask, batch_first=True)
        venus_mask = pad_sequence(venus_mask, batch_first=True)
        server_model = torch.tensor(list(data.iloc[:].server_model))
        crash_dump = torch.tensor([json.loads(line) for line in list(data.iloc[:].crashdump_feature)],
                                  dtype=torch.int)

        data = []
        # self.data = torch.stack([msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump], dim=0)
        for q, w, e, r, t, y in zip(msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump):
            data.append([q, w, e, r, t, y])
        self.data = data

        label = np.array(label)
        label_key = sorted(np.unique(label))

        encode_labels = np.arange(len(label_key))
        label_map = dict(zip(label_key, encode_labels))
        label = np.array([label_map[lab] for lab in label])

        self.root_path = root_path
        self.split_tag = split_tag
        self.label = label
        self.n_classes = len(label_key)

        self.base = base
        self.novel = novel

        if normalize:
            pass

        self.transform = transform


class AIOPSVal(AIOPSTest):
    def __init__(self, root_path, split="val", base=None, novel=None):
        super().__init__(root_path, split=split, base=base, novel=novel)


class MetaAIOPS(Dataset):
    def __init__(self, root_path, split="train", normalize=False,
                 transform=None, val_transform=None,
                 n_batch=200, n_episode=4, n_way=2, n_shot=1, n_query=15, novel_class=None,
                 mode="train", finetune_class=None,
                 ):
        super(MetaAIOPS, self).__init__()

        if novel_class is None:
            novel_class = [3, ]

        split_dict = {'train': 'train_set',  # standard train
                      'val': 'a_test_set',
                      'test': 'b_test_set',
                      'meta-train': 'train_set',  # meta-train
                      'meta-val': 'a_test_set',  # meta-val
                      'meta-test': 'b_test_set',  # meta-test
                      }

        split_tag = split_dict[split]
        split_file_path = os.path.join(root_path, split_tag + '.csv')
        assert os.path.exists(split_file_path)
        split_file = pd.read_csv(split_file_path)

        data = split_file.drop("label", axis=1)
        label = split_file['label']

        msg_all, msg_mask = [], []
        venus_all, venus_mask = [], []
        msg_sentence_max, venus_sentence_max = 0, 0

        for sample in list(data.iloc[:].msg_feature.values):
            msgs = json.loads(sample)
            msg_all.append(torch.tensor(msgs))
            msg_mask.append(torch.ones(len(msgs)))
            msg_sentence_max = max(msg_sentence_max, len(msgs))

        for sample in list(data.iloc[:].venus_feature.values):
            venus = json.loads(sample)
            venus_all.append(torch.tensor(venus))
            venus_mask.append(torch.ones(len(venus)))
            venus_sentence_max = max(venus_sentence_max, len(venus))

        msg_all = np.array([F.pad(sample, (0, 0, 0, msg_sentence_max - sample.shape[0])).numpy() for sample in msg_all])
        msg_all = torch.tensor(msg_all)
        venus_all = pad_sequence(venus_all, batch_first=True)
        msg_mask = pad_sequence(msg_mask, batch_first=True)
        venus_mask = pad_sequence(venus_mask, batch_first=True)
        server_model = torch.tensor(list(data.iloc[:].server_model))
        crash_dump = torch.tensor([json.loads(line) for line in list(data.iloc[:].crashdump_feature)],
                                  dtype=torch.int)

        data = []
        # self.data = torch.stack([msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump], dim=0)
        for q, w, e, r, t, y in zip(msg_all, msg_mask, venus_all, venus_mask, server_model, crash_dump):
            data.append([q, w, e, r, t, y])
        self.data = data
        label = np.array(label)
        label_key = sorted(np.unique(label))

        self.root_path = root_path
        self.split_tag = split_tag
        self.label = label
        self.n_classes = len(label_key)

        if normalize:
            pass
        self.transform = transform
        self.val_transform = val_transform

        # -----------------------------------------------------------------
        # meta load

        assert mode in ["train", "finetune"]
        self.mode = mode
        if self.mode == "finetune":
            if finetune_class is None:
                finetune_class = [0, ]

            set1 = set(finetune_class)
            set2 = set(novel_class)
            if set1 & set2:
                raise ValueError("finetune class must be different from novel class!")
            del set1, set2

            finetune_class = finetune_class + novel_class

        self.finetune_class = finetune_class
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_query = n_query
        self.n_shot = n_shot

        self.novel_class = novel_class
        self.base_class = [i for i in range(self.n_classes) if i not in self.novel_class]

        self.cat_locs = tuple()
        for cat in range(self.n_classes):
            self.cat_locs += (np.argwhere(self.label == cat).reshape(-1),)

    def __len__(self):
        if self.mode == "train":
            return self.n_batch * len(self.base_class)
        elif self.mode == "finetune":
            return 1
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        shot, query = [], []
        assert self.n_way <= len(self.base_class)
        if self.mode == "train":
            cats = np.random.choice(self.base_class, self.n_way, replace=False)
        elif self.mode == "finetune":
            # cats = np.random.choice(self.base_class, self.n_way - 1, replace=False)
            cats = self.finetune_class
        else:
            raise NotImplementedError

        for c in cats:
            c_shots, c_query = [], []
            idx_list = np.random.choice(
                self.cat_locs[c], self.n_shot + self.n_query, replace=False)
            shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
            for idx in shot_idx:
                c_shots.append(self.data[idx])
            for idx in query_idx:
                c_query.append(self.data[idx])

            shot.append(c_shots)
            query.append(c_query)

        # shot = torch.stack(shot, dim=0)
        # query = torch.stack(query, dim=0)
        shot = [l for li in shot for l in li]
        query = [l for li in query for l in li]
        cls = torch.arange(self.n_way)[:, None]

        shot_labels = cls.repeat(1, self.n_shot).flatten()  # [n_way * n_shot]
        query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]

        return shot, query, shot_labels, query_labels


if __name__ == '__main__':
    ai = AIOPSTest(root_path="./aliyunwei/tmp_data", split="train", base=[0, 1], novel=[2])
    x = ai[0]
    print()
