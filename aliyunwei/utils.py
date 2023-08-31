import pandas as pd
import datetime as dt
from collections import Counter


#
# train_set = pd.read_csv("./tmp_data/train_set.csv")
#
# test_seta = pd.read_csv("./tmp_data/test_set_a.csv")
# test_seta_label = pd.read_csv("./data/test_ab/preliminary_test_label_dataset_a.csv")
# test_seta = test_seta.merge(test_seta_label, on=['sn', 'fault_time'], how='inner')
# test_seta = test_seta.drop(labels='positive_p', axis=1)
# test_seta.to_csv("./tmp_data/a_test_set.csv", index=False)
#
# test_setb = pd.read_csv("./tmp_data/test_set_b.csv")
# test_setb_label = pd.read_csv("./data/test_ab/preliminary_test_label_dataset_b.csv")
# test_setb = test_setb.merge(test_setb_label, on=['sn', 'fault_time'], how='inner')
# test_setb = test_setb.drop(labels='positive_p', axis=1)
# test_setb.to_csv("./tmp_data/b_test_set.csv", index=False)
def move_to_device(nested_list, device):
    if isinstance(nested_list[0], list):
        # 如果当前元素是列表，递归调用 move_to_device 函数
        nested_tensors = [move_to_device(sublist, device=device) for sublist in nested_list]
    else:
        # 如果当前元素是张量，添加一个虚拟的维度，并将它放到指定的设备上
        nested_tensors = [tensor.unsqueeze(0).to(device) if tensor.dim() == 0 else tensor.to(device) for tensor in
                          nested_list]
    return nested_tensors


class Logger:
    def __init__(self, log_path, end="\n", stdout=True):
        self.log_path = log_path
        self.end = end
        self.stdout = stdout

    def log(self, message):
        time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
        log_str = "[{}] {}{}".format(time_str, message, self.end)
        with open(self.log_path, "a") as f:
            f.write(log_str)
        if self.stdout:
            print(log_str, end="")


def print_args(parser, args, only_non_defaut=False):
    default_str_list = ['=====default args=====']
    non_default_str_list = ['=====not default args=====']

    args_dict = vars(args)
    for k, v in args_dict.items():
        default = parser.get_default(k)
        if v == default:
            default_str_list.append('{}: {}'.format(k, v))
        else:
            non_default_str_list.append('{}: {} (default: {})'.format(k, v, default))

    default_str = '\n'.join(default_str_list)
    non_default_str = '\n'.join(non_default_str_list)

    print(non_default_str)
    if not only_non_defaut:
        print(default_str)
    print('-' * 15)


def count_label(dataloader):
    labels = []
    for data in dataloader:
        _, label = data
        for i in list(label):
            labels.append(i.item())

    label_counts = Counter(labels)
    print(label_counts)
