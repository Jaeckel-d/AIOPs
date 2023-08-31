import os
from collections import OrderedDict
import torch
import torch.nn as nn

# import sys
# sys.path.insert(0, sys.path[0] + "/../../")
# print(sys.path)

from torch.nn import functional as F
import math
import json

interval_bucket_num, cnt_bucket_num, duration_bucket_num = 101, 23, 12
lookup1 = torch.load("/home/wyd/AIOP/aliyunwei/tmp_data/word2idx.pk")
venus_dict = json.load(open('/home/wyd/AIOP/aliyunwei/tmp_data/venus_dict.json', 'r'))
crash_dump_dict = json.load(open('/home/wyd/AIOP/aliyunwei/tmp_data/crashdump_dict.json', 'r'))

__all__ = ["AttentionLSTM",
           "AttentionPool",
           "GateAttention",
           "AttentionPool_"]


class AttentionPooling1D(nn.Module):
    """通过加性Attention,将向量序列融合为一个定长向量
    """

    def __init__(self, in_features):
        super(AttentionPooling1D, self).__init__()
        self.in_features = in_features  # 词向量维度
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)

    def forward(self, inputs):
        xo, mask = inputs
        mask = mask.unsqueeze(-1)
        x = self.k_dense(xo)
        x = self.o_dense(torch.tanh(x))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2)  # N*emd


class AttentionForLSTM(nn.Module):
    """通过加性Attention,将向量序列融合为一个定长向量
    """

    def __init__(self, in_features):
        super(AttentionForLSTM, self).__init__()
        assert in_features % 2 == 0, 'in_features must be even'
        self.in_features = in_features  # 词向量维度
        # self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.lstm = nn.LSTM(in_features, in_features // 2, batch_first=True, bidirectional=True)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)

    def forward(self, inputs):
        xo, mask = inputs
        xo, _ = self.lstm(xo)
        mask = mask.unsqueeze(-1)
        x = self.o_dense(torch.tanh(xo))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2)  # N*emd


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Returns: [seq_len, d_hid]
    """
    embeddings_table = torch.zeros(n_position, d_hid)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class ScalaOffset(nn.Module):
    def __init__(self, dim=10) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.offset = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, _input):
        assert self.dim == _input.shape[-1], "dim 维度不对"
        out = _input * self.scale + self.offset
        return out


class GateAttentionUnit(nn.Module):
    def __init__(self, unit=46, key_size=16) -> None:
        super().__init__()
        self.unit = unit
        self.key_size = key_size
        self.i_dense = nn.Linear(unit, 2 * unit + key_size)
        self.o_dense = nn.Linear(unit, unit)
        self.scale_offset_q = ScalaOffset(dim=key_size)
        self.scale_offset_k = ScalaOffset(dim=key_size)
        self.query = nn.Linear(unit, 1)
        self.rope = RoPEPositionEncoding(50, 16)

    def forward(self, _input):
        x, mask = _input
        mask_trans = mask[:, None, :]
        u, v, qk = torch.split(F.silu(self.i_dense(x)), [self.unit, self.unit, self.key_size], dim=-1)
        q, k = self.scale_offset_q(qk), self.scale_offset_k(qk)
        q, k = self.rope(q), self.rope(k)
        a = torch.matmul(q, k.transpose(1, 2)) / (self.key_size ** 0.5)
        a = torch.softmax(a * mask_trans + (1 - mask_trans) * -1e12, dim=-1)
        out = torch.matmul(a, v) * u
        out = torch.matmul(a, v)
        out = self.o_dense(out)
        prob = self.query(out).squeeze()  #
        prob = prob * mask - (1 - mask) * 1e12
        prob = torch.softmax(prob, dim=-1)
        return torch.sum(out * prob.unsqueeze(dim=-1), dim=1)


class AttentionLSTM(nn.Module):
    def __init__(self, features=26):
        super(AttentionLSTM, self).__init__()
        # msg embedding
        self.emb_msg_f1 = nn.Embedding(len(lookup1), 12, padding_idx=0)
        self.emb_msg_f2 = nn.Embedding(len(lookup1), 6, padding_idx=0)  # 152个
        self.emb_msg_f3 = nn.Embedding(len(lookup1), 8, padding_idx=0)  # 498

        # else
        self.emd_server_model = nn.Embedding(88, 8, padding_idx=0)
        self.emb_venus = nn.Embedding(len(venus_dict), 4, padding_idx=0)
        # self.attention = AttentionForLSTM(features)

        self.out_dim = 12 + 6 + 8 + 8 + 4 * 3
        self.classify = nn.Linear(self.out_dim, 10)
        self.dense2 = nn.Linear(10, 4)

    def forward(self, features, params=None):
        if params is None:
            params = list(self.parameters())

        msgs, msg_mask, venus_batch, venus_mask, server_model, crash_dump = features  # len1 batch_size * sentence_num

        # msg_f1 = self.emb_msg_f1(msgs[..., 0])  # (b, s, 3, d)
        # msg_f2 = self.emb_msg_f2(msgs[..., 1], params[1])
        # msg_f3 = self.emb_msg_f3(msgs[..., 2], params[2])
        msg_f1 = F.embedding(msgs[..., 0], weight=params[0])
        msg_f2 = F.embedding(msgs[..., 1], weight=params[1])
        msg_f3 = F.embedding(msgs[..., 2], weight=params[2])

        msg_emb = torch.concat([msg_f1, msg_f2, msg_f3], dim=-1)
        att_emd = self.attention((msg_emb, msg_mask))
        msg_mask = msg_mask.unsqueeze(dim=-1)

        venus = self.emb_venus(venus_batch)
        b, s, n, d = venus.shape
        venus = venus.view(b, s, -1)
        venus = torch.sum(venus * venus_mask.unsqueeze(dim=-1), dim=1)

        server_model = self.emd_server_model(server_model)
        out = torch.concat([att_emd, server_model, venus], dim=-1)

        score = self.classify(out)
        score = self.dense2(torch.relu(score))
        return score

    def get_out_dim(self):
        return self.out_dim


class GateAttention(nn.Module):
    def __init__(self, features=34):
        super(GateAttention, self).__init__()
        # msg embedding
        self.emb_msg_f1 = nn.Embedding(len(lookup1), 12, padding_idx=0)
        self.emb_msg_f2 = nn.Embedding(len(lookup1), 6, padding_idx=0)  # 152个
        self.emb_msg_f3 = nn.Embedding(len(lookup1), 8, padding_idx=0)  # 498

        # else
        self.emd_server_model = nn.Embedding(88, 8, padding_idx=0)
        self.emb_venus = nn.Embedding(len(venus_dict), 4, padding_idx=0)
        self.attention = GateAttention(features)

        self.out_dim = 12 + 6 + 8 + 8 + 4 * 3
        self.classify = nn.Linear(self.out_dim, 10)
        self.dense2 = nn.Linear(10, 4)

    def forward(self, features):
        msgs, msg_mask, venus_batch, venus_mask, server_model, crash_dump = features  # len1 batch_size * sentence_num

        msg_f1 = self.emb_msg_f1(msgs[..., 0])  # (b, s, 3, d)
        msg_f2 = self.emb_msg_f2(msgs[..., 1])
        msg_f3 = self.emb_msg_f3(msgs[..., 2])
        msg_emb = torch.concat([msg_f1, msg_f2, msg_f3], dim=-1)
        att_emd = self.attention((msg_emb, msg_mask))

        venus = self.emb_venus(venus_batch)
        b, s, n, d = venus.shape
        venus = venus.view(b, s, -1)
        venus = torch.sum(venus * venus_mask.unsqueeze(dim=-1), dim=1)

        server_model = self.emd_server_model(server_model)
        out = torch.concat([att_emd, server_model, venus], dim=-1)

        score = self.classify(out)
        score = self.dense2(torch.relu(score))
        return score


class AttentionPool_(nn.Module):
    def __init__(self, features=26, binary=False):
        super(AttentionPool_, self).__init__()
        # msg embedding
        self.emb_msg_f1 = nn.Embedding(len(lookup1), 12, padding_idx=0)
        self.emb_msg_f2 = nn.Embedding(len(lookup1), 6, padding_idx=0)  # 152个
        self.emb_msg_f3 = nn.Embedding(len(lookup1), 8, padding_idx=0)  # 498

        # else
        self.emd_server_model = nn.Embedding(88, 8, padding_idx=0)
        self.emb_venus = nn.Embedding(len(venus_dict), 4, padding_idx=0)
        self.attention = AttentionPooling1D(features)

        self.out_dim = 12 + 6 + 8 + 8 + 4 * 3
        self.classify = nn.Linear(self.out_dim, 10)
        self.binary = binary
        if self.binary:
            self.dense2 = nn.Linear(10, 1)
        else:
            self.dense2 = nn.Linear(10, 3)

    def forward(self, features):
        msgs, msg_mask, venus_batch, venus_mask, server_model, crash_dump = features  # len1 batch_size * sentence_num

        msg_f1 = self.emb_msg_f1(msgs[..., 0])  # (b, s, 3, d)
        msg_f2 = self.emb_msg_f2(msgs[..., 1])
        msg_f3 = self.emb_msg_f3(msgs[..., 2])
        msg_emb = torch.concat([msg_f1, msg_f2, msg_f3], dim=-1)
        att_emd = self.attention((msg_emb, msg_mask))

        venus = self.emb_venus(venus_batch)
        b, s, n, d = venus.shape
        venus = venus.view(b, s, -1)
        venus = torch.sum(venus * venus_mask.unsqueeze(dim=-1), dim=1)

        server_model = self.emd_server_model(server_model)
        out = torch.concat([att_emd, server_model, venus], dim=-1)

        score = self.classify(out)
        score = self.dense2(torch.relu(score))
        # return torch.sigmoid(score)
        return score


class AttentionPool(nn.Module):
    def __init__(self, features=26):
        super(AttentionPool, self).__init__()
        # msg embedding
        self.emb_msg_f1 = nn.Embedding(len(lookup1), 12, padding_idx=0)
        self.emb_msg_f2 = nn.Embedding(len(lookup1), 6, padding_idx=0)  # 152个
        self.emb_msg_f3 = nn.Embedding(len(lookup1), 8, padding_idx=0)  # 498

        # else
        self.emd_server_model = nn.Embedding(88, 8, padding_idx=0)
        self.emb_venus = nn.Embedding(len(venus_dict), 4, padding_idx=0)
        # self.attention = AttentionPooling1D(features)
        self.in_features = features
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)

        self.out_dim = 12 + 6 + 8 + 8 + 4 * 3
        self.classify = nn.Linear(self.out_dim, 10)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, features, params=None):
        if params is None:
            params = list(self.parameters())

        msgs, msg_mask, venus_batch, venus_mask, server_model, crash_dump = features  # len1 batch_size * sentence_num

        # msg_f1 = self.emb_msg_f1(msgs[..., 0])  # (b, s, 3, d)
        # msg_f2 = self.emb_msg_f2(msgs[..., 1], params[1])
        # msg_f3 = self.emb_msg_f3(msgs[..., 2], params[2])
        msg_f1 = F.embedding(msgs[..., 0], weight=params[0])
        msg_f2 = F.embedding(msgs[..., 1], weight=params[1])
        msg_f3 = F.embedding(msgs[..., 2], weight=params[2])

        msg_emb = torch.concat([msg_f1, msg_f2, msg_f3], dim=-1)
        # att_emd = self.attention((msg_emb, msg_mask))
        msg_mask = msg_mask.unsqueeze(-1)
        x = F.linear(msg_emb, params[5])
        x = F.linear(torch.tanh(x), params[6])
        x = x - (1 - msg_mask) * 1e12
        x = F.softmax(x, dim=-2)
        att_emd = torch.sum(x * msg_emb, dim=-2)

        # venus = self.emb_venus(venus_batch)
        venus = F.embedding(venus_batch, weight=params[4])
        b, s, n, d = venus.shape
        venus = venus.view(b, s, -1)
        venus = torch.sum(venus * venus_mask.unsqueeze(dim=-1), dim=1)

        # server_model = self.emd_server_model(server_model)
        server_model = F.embedding(server_model, weight=params[3])

        out = torch.concat([att_emd, server_model, venus], dim=-1)

        # score = self.classify(out)
        score = F.linear(out, params[7], bias=params[8])
        # score = self.dense2(torch.relu(score))
        score = F.linear(torch.relu(score), params[9], params[10])
        # return torch.sigmoid(score)
        return score

    def get_out_dim(self):
        return self.out.shape[2]

# if __name__ == "__main__":
#     ai = AIOPS(root_path="/home/wyd/AIOP/aliyunwei/tmp_data", split="train")
#     test = AttentionLSTM()
#     print(test(ai[0][0]))
