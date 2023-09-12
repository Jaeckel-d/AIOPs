# MAML and FOMAML training and finetune

from maml_ import MAML
from model import AttentionPool, AttentionLSTM, GateAttention
from aiop_dataset import MetaAIOPS
from torch.utils.data.dataloader import DataLoader
import torch
import os

# Dataset hyperparameters
num_outer_steps = 15000
batch_size = 4
num_train_tasks = batch_size * num_outer_steps
num_shots_support = 10
num_shots_query = 10
num_test_tasks = 1000

# Model hyperparameters
num_inner_steps = 1
inner_lr = 1e-2
outer_lr = 1e-3
log_interval = 50
num_inner_steps_test = 10

# Collect data
# dataset = SinusoidDataset()
# data_train = dataset.get_data(batch_size, num_shots_support,
#                               num_shots_query, num_train_tasks)
# data_test = dataset.get_data(1, num_shots_support, num_shots_query,
#                              num_test_tasks)

# Initialize MAML, FOMAML, and baseline


data_train = MetaAIOPS(root_path="./aliyunwei/tmp_data", mode="train", n_shot=10)
data_train = DataLoader(data_train, batch_size=batch_size)

# data_test = dataset.get_data(1, num_shots_support, num_shots_query,
#                              num_test_tasks)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionPool()
maml = MAML(model, num_inner_steps, inner_lr, outer_lr, False, epochs=300, device=device)
fomaml = MAML(model, num_inner_steps, inner_lr, outer_lr, True, epochs=300, device=device)

# Train and test models
# models = [maml, fomaml]
# names = ["MAML", "FOMAML"]


# for model, name in zip(models, names):
#     # model.train(data_train, log_interval, f"./logs/{name}_train.json")
#     repeat = 90
#     for i in range(repeat):
#         data_test = MetaAIOPS(root_path="./aliyunwei/tmp_data", mode="finetune", n_shot=n_shot,
#                               finetune_class=finetune_class)
#         data_test = DataLoader(data_test, batch_size=1, shuffle=True)
#
#         dir_path = os.path.join(root_path, f"{name}_{n_shot}_shot_finetuneclass_{finetune_class[0]}")
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         save_path = os.path.join(dir_path, f"{name}_{n_shot}_shot_{finetune_class}_finetune_class_{i + 10}.pt")
#         model.finetune_(data_test, 30,
#                         load_path="/home/wyd/AIOP/MAML/meta-training/fomaml_10_shot_2023-07-16-14_50_12_ckpt.pt",
#                         save_path=save_path
#                         )
#     # model.test(data_test, num_inner_steps_test)

load_paths = [
    # "/home/wyd/AIOP/MAML/meta-training/1_shot_2023-07-13-21_20_39_ckpt.pt",
    # "/home/wyd/AIOP/MAML/meta-training/5_shot_2023-07-13-21_21_17_ckpt.pt",
    # "/home/wyd/AIOP/MAML/meta-training/10_shot_2023-07-13-21_22_47_ckpt.pt",
    "/home/wyd/AIOP/MAML/meta-training/fomaml_1_shot_2023-07-13-21_20_39_ckpt.pt",
    "/home/wyd/AIOP/MAML/meta-training/fomaml_5_shot_2023-07-13-21_21_17_ckpt.pt",
    "/home/wyd/AIOP/MAML/meta-training/fomaml_10_shot_2023-07-16-14_50_12_ckpt.pt"
]

finetune_class = [[0, ], [1, ], [2, ]]
root_path = "/home/wyd/AIOP/MAML/meta-finetune"
models = [fomaml]
names = ["FOMAML"]

if __name__ == "__main__":
    for finetune in finetune_class:
        for x, shot in enumerate([1, 5, 10]):
            for model, name in zip(models, names):
                repeat = 90
                for i in range(repeat):
                    data_test = MetaAIOPS(root_path="./aliyunwei/tmp_data", mode="finetune", n_shot=shot,
                                          finetune_class=finetune)
                    data_test = DataLoader(data_test, batch_size=1, shuffle=True)

                    dir_path = os.path.join(root_path, f"{name}_{shot}_shot_finetuneclass_{finetune[0]}")
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)
                    save_path = os.path.join(dir_path, f"{name}_{shot}_shot_{finetune}_finetune_class_{i + 10}.pt")
                    model.finetune_(data_test, 30,
                                    load_path=load_paths[x],
                                    save_path=save_path
                                    )
            # model.test(data_test, num_inner_steps_test)
