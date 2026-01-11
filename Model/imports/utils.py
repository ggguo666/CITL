import os
import random
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from imports import preprocess_data as Reader
def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()


def train_val_test_split_ABIDEII_ABIDE(kfold = 10, fold = 0, seed = 0):
    n_sub = 211586
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id
def train_val_test_split_ABIDE_ADHD(kfold = 10, fold = 0, seed = 0):
    n_sub = 252633
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id


def train_val_test_split_stage3_ABIDE(kfold = 10, fold = 0, seed = 0):
# def train_val_test_split_stage3_ABIDE(kfold=10, fold=0):
    n_sub = 1035
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id


def train_val_test_split_stage3_ADHD(kfold = 10, fold = 0, seed = 0):
# def train_val_test_split_stage3_ADHD(kfold=10, fold=0):
    n_sub = 782
    id = list(range(n_sub))
    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id

def train_val_test_split_fold(data_folder, fold, kfold=10):
    # random_seed = np.random.randint(0, 100)
    random_seed = 41
    print("数据划分种子点", random_seed)
    # sorted_files = Reader.get_ids_dfc(data_folder)###abdie2使用
    # print(sorted_files)
    # # 获取数据文件列表
    files = sorted(os.listdir(data_folder))
    print(len(files))
    sorted_files = sorted(files, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))

    # 创建字典，用于存储前缀及其对应的文件列表
    prefixes = {}
    for file_idx, file in enumerate(sorted_files):
        prefix = file.split('-')[0]
        suffix = file.split('-')[1]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append((file_idx, suffix))
    # print(prefixes)
    train_files = []
    val_files = []
    test_files = []

    prefixes_list = list(prefixes.keys())  # 获取前缀列表
    random.seed(random_seed)
    random.shuffle(prefixes_list)




    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(prefixes_list)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    train_prefixes = [prefixes_list[i] for i in train_id if i < len(prefixes_list)]
    val_prefixes = [prefixes_list[i] for i in val_id if i < len(prefixes_list)]
    test_prefixes = [prefixes_list[i] for i in test_id if i < len(prefixes_list)]
    # train_prefixes = prefixes_list[train_id]
    # val_prefixes = prefixes_list[val_id]
    # test_prefixes = prefixes_list[test_id]

    # print(train_prefixes)
    # print(val_prefixes)
    # print(test_prefixes)
    # 将后缀的文件索引添加到其对应的前缀数据集中
    for prefix in train_prefixes:
        train_files.extend(prefixes[prefix])
        # print(train_files)

    for prefix in val_prefixes:
        val_files.extend(prefixes[prefix])
        # print(val_files)

    for prefix in test_prefixes:
        test_files.extend(prefixes[prefix])
        # print(test_files)

    train_index = [item[0] for item in train_files]
    val_index = [item[0] for item in val_files]
    test_index = [item[0] for item in test_files]

    return train_index, val_index, test_index


def train_val_test_split(data_folder, val_ratio=0.2, test_ratio=0.1):
    # random_seed = np.random.randint(0, 100)
    random_seed = 41
    print("数据划分种子点", random_seed)
    sorted_files = Reader.get_ids_abide2(data_folder)
    # print(sorted_files)
    # # 获取数据文件列表
    # files = sorted(os.listdir(data_folder))
    # print(len(files))
    # sorted_files = sorted(files, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))

    # 创建字典，用于存储前缀及其对应的文件列表
    prefixes = {}
    for file_idx, file in enumerate(sorted_files):
        prefix = file.split('-')[0]
        suffix = file.split('-')[1]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append((file_idx, suffix))
    # print(prefixes)
    train_files = []
    val_files = []
    test_files = []

    prefixes_list = list(prefixes.keys())  # 获取前缀列表
    random.seed(random_seed)
    random.shuffle(prefixes_list)
    split_val_idx = int(len(prefixes_list) * (1 - val_ratio - test_ratio))
    split_test_idx = int(len(prefixes_list) * (1 - test_ratio))
    train_prefixes = prefixes_list[:split_val_idx]
    val_prefixes = prefixes_list[split_val_idx:split_test_idx]
    test_prefixes = prefixes_list[split_test_idx:]

    # 将后缀的文件索引添加到其对应的前缀数据集中
    for prefix in train_prefixes:
        train_files.extend(prefixes[prefix])
        # print(train_files)

    for prefix in val_prefixes:
        val_files.extend(prefixes[prefix])
        # print(val_files)

    for prefix in test_prefixes:
        test_files.extend(prefixes[prefix])
        # print(test_files)

    train_index = [item[0] for item in train_files]
    val_index = [item[0] for item in val_files]
    test_index = [item[0] for item in test_files]

    return train_index, val_index, test_index


def train_val_test_split_stage3(kfold = 10, fold = 0):
    n_sub = 1035
    id = list(range(n_sub))


    import random
    random.seed(77)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id

def train_val_test_split_basemodel(kfold = 10, fold = 0, seed = 0):
    n_sub = 1035
    id = list(range(n_sub))


    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)
    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id

def adhd_train_val_test_split_basemodel(kfold = 10, fold = 0,seed=0):
    n_sub = 782
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id

def train_val_test():
    # n_sub = 1035
    n_sub = 782
    id = list(range(n_sub))

    # Generate a random seed
    random_state = np.random.randint(0, 100)
    # random_state = 87
    print("数据划分种子点", random_state)
    # Split into 70% train and 30% test
    train_ids, test_ids = train_test_split(id, test_size=0.3, random_state=random_state)

    # Further split test set into 20% validation and 10% test
    val_ids, test_ids = train_test_split(test_ids, test_size=1/3, random_state=random_state)

    return train_ids, val_ids, test_ids

def train_test_split_stage3(kfold=10, fold=0, seed = 0):
    n_sub = 1035
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)

    train_index = list()
    test_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        train_index.append(tr)

    train_id = train_index[fold]
    test_id = test_index[fold]
    # print(train_id)
    # print(test_id)
    return train_id, test_id


def train_test_split_stage3_adhd(kfold=10, fold=0, seed = 0):
    n_sub = 782
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)

    train_index = list()
    test_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        train_index.append(tr)

    train_id = train_index[fold]
    test_id = test_index[fold]
    # print(train_id)
    # print(test_id)
    return train_id, test_id


def train_test_split_stage3_abide2(kfold=10, fold=0, seed = 0):
    n_sub = 586
    id = list(range(n_sub))

    # seed = np.random.randint(0, 100)
    # print("数据划分种子点", seed)
    random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, shuffle=True)

    train_index = list()
    test_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        train_index.append(tr)

    train_id = train_index[fold]
    test_id = test_index[fold]
    # print(train_id)
    # print(test_id)
    return train_id, test_id

#
# # 使用示例
# data_folder = "F:\\PyCharm 2020.3.5\\新建文件夹\\BrainGNN_Pytorch-main\\BrainGNN_Pytorch-main\\data\\ABIDE_dfc\\cpac\\filt_noglobal\\raw"
# train_index, val_index, test_index = train_val_test_split(data_folder, val_ratio=0.1, test_ratio=0.2)

