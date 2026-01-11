import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from imports import preprocess_data as Reader
import pandas as pd
import numpy as np
import itertools
import os
import glob
import re
def sort_paths_by_last_number(paths):
    # 过滤出以'sub-'开头的路径，然后按照数字大小排序
    filtered_paths = [path for path in paths if re.search(r'sub-(\d+)$', path)]
    sorted_paths = sorted(filtered_paths, key=lambda x: int(re.search(r'sub-(\d+)$', x).group(1)))
    return sorted_paths

# 自定义排序函数，按照路径中最后一部分的数字进行排序
def ADHD_sort_by_last_number(path):
    filename = os.path.basename(path)  # 获取文件名部分
    number_str = os.path.splitext(filename)[0]  # 去掉扩展名，获取纯数字部分
    return int(number_str)  # 将纯数字部分转换为整数，以便排序
class Simialr_ASD_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        self.site_feature = {}
        FC_list = []
        label_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc_similar(FC_dict)
        print(file_list)
        label_list = Reader.get_label_dfc_similar(file_list, score='Label')

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)

        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)

            subject_list = [file for file in subject_list if file[:-4].isdigit()]
            sorted_filenames = Reader.sort_filenames(subject_list)
            # print(sorted_filenames)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label


class ASD_abide2_Dataset_official(Dataset):
    def __init__(self, FC_dict1, FC_dict2, save_dir=None):
        self.site_feature = {}
        FC_list = []
        FC_list2 = []
        label_list = []
        label_list2 =[]
        merged_label_list=[]
        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(
                    os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回





        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc_abide_adhd(FC_dict1)

        print(len(file_list))
        file_list2 = Reader.get_ids_abide2(FC_dict2)
        # print(file_list2)
        print(len(file_list2))
        merged_list = list(itertools.chain(file_list, file_list2))
        # print(merged_list)
        print(len(merged_list))
        label_list = Reader.get_label_dfc_abide_adhd(file_list, score='DX_GROUP')
        print(len(label_list))
        label_list2 = Reader.get_abide2_dfc(file_list2, score='DX_GROUP')
        print(len(label_list2))
        merged_label_list = list(itertools.chain(label_list, label_list2))

        # print(merged_label_list)
        print(len(merged_label_list))

        file_paths = glob.glob(os.path.join(FC_dict1, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)

        file_paths2 = glob.glob(os.path.join(FC_dict2, '*'))
        sorted_paths2 = sort_paths_by_last_number(file_paths2)

        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            # print(subject_list)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            sorted_filenames = Reader.sort_filenames(subject_list)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        for file_path in sorted_paths2:
            subject_list = os.listdir(file_path)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            file_paths_sorted = sorted(subject_list, key=sort_by_number)

            print(file_paths_sorted)
            for subject_id in file_paths_sorted:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list2.append(matrix)


        merged_FC_list = list(itertools.chain(FC_list, FC_list2))

        self.FCs = merged_FC_list
        self.labels = merged_label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label



class ASD_ADHD_Dataset_official(Dataset):
    def __init__(self, FC_dict1, FC_dict2, save_dir=None):
        self.site_feature = {}
        FC_list = []
        FC_list2 = []
        label_list = []
        label_list2 =[]
        merged_label_list=[]
        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(
                    os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc_abide_adhd(FC_dict1)
        print(len(file_list))
        file_list2 = Reader.get_ids_ADHD(FC_dict2)
        # print(file_list2)
        print(len(file_list2))
        merged_list = list(itertools.chain(file_list, file_list2))
        # print(merged_list)
        print(len(merged_list))
        label_list = Reader.get_label_dfc_abide_adhd(file_list, score='DX_GROUP')
        print(len(label_list))
        # print(len(label_list))
        label_list2 = Reader.get_subject_score_ADHD(file_list2, score='DX')
        print(len(label_list2))
        merged_label_list = list(itertools.chain(label_list, label_list2))

        # print(merged_label_list)
        print(len(merged_label_list))

        file_paths = glob.glob(os.path.join(FC_dict1, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)

        file_paths2 = glob.glob(os.path.join(FC_dict2, '*'))
        sorted_paths2 = sort_paths_by_last_number(file_paths2)

        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            # print(subject_list)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            sorted_filenames = Reader.sort_filenames(subject_list)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        for file_path in sorted_paths2:
            subject_list = os.listdir(file_path)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            file_paths_sorted = sorted(subject_list, key=sort_by_number)

            print(file_paths_sorted)
            for subject_id in file_paths_sorted:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list2.append(matrix)


        merged_FC_list = list(itertools.chain(FC_list, FC_list2))

        self.FCs = merged_FC_list
        self.labels = merged_label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label

class ADHD_ASD_Dataset_official(Dataset):
    def __init__(self, FC_dict1, FC_dict2, save_dir=None):
        self.site_feature = {}
        FC_list = []
        FC_list2 = []
        label_list = []
        label_list2 =[]
        merged_label_list=[]
        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(
                    os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回




        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc_abide_adhd2(FC_dict2)
        # print(file_list)
        print(len(file_list))
        file_list2 = Reader.get_ids_dfc(FC_dict1)
        # print(file_list2)
        print(len(file_list2))
        merged_list = list(itertools.chain(file_list, file_list2))
        # print(merged_list)
        print(len(merged_list))
        label_list = Reader.get_label_dfc_abide_adhd2(file_list, score='DX')
        print(len(label_list))
        # print(len(label_list))
        label_list2 = Reader.get_label_dfc(file_list2, score='DX_GROUP')
        # print(label_list2)
        print(len(label_list2))
        merged_label_list = list(itertools.chain(label_list, label_list2))

        # print(merged_label_list)
        print(len(merged_label_list))


        file_paths = glob.glob(os.path.join(FC_dict2, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)
        # print(sorted_paths)
        file_paths2 = glob.glob(os.path.join(FC_dict1, '*'))
        sorted_paths2 = sort_paths_by_last_number(file_paths2)

        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            # print(subject_list)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            sorted_filenames = Reader.sort_filenames(subject_list)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        for file_path in sorted_paths2:
            subject_list = os.listdir(file_path)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            file_paths_sorted = sorted(subject_list, key=sort_by_number)

            # print(file_paths_sorted)
            for subject_id in file_paths_sorted:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list2.append(matrix)


        merged_FC_list = list(itertools.chain(FC_list, FC_list2))

        self.FCs = merged_FC_list
        self.labels = merged_label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label

class ASD_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        self.site_feature = {}
        FC_list = []
        label_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_dfc(FC_dict)
        print(file_list)
        label_list = Reader.get_label_dfc(file_list, score='DX_GROUP')
        print(label_list)
        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)

        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            sorted_filenames = Reader.sort_filenames(subject_list)
            for subject_id in sorted_filenames:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label

# 定义一个函数，从路径中提取数字
def extract_number(path):
    # 使用正则表达式提取最后一个数字
    match = re.search(r'sub-(\d+)\.npy$', path)
    if match:
        return int(match.group(1))  # 提取括号中的数字部分
    else:
        return 0


class fc_ASD_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []


        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids(FC_dict)
        label_list = Reader.get_subject_score(file_list, score='DX_GROUP')

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        # print(file_paths)
        sorted_paths = sorted(file_paths, key=lambda x: extract_number(x))
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            print(file_path)
            matrix = np.load(file_path)
            FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label



class stage2_abide_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):

        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sorted(file_paths, key=ADHD_sort_by_last_number)
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            print(file_path)
            matrix = np.load(file_path)
            FC_list.append(matrix)

        self.FCs = FC_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.FCs)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        return data


class abide_ave_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ave(FC_dict)
        print(len(file_list))
        label_list = Reader.get_ABIDE_subject_score_ave(file_list, score='DX_GROUP')
        print(len(label_list))
        for subject_id in file_list:
            # print(subject_id)
            path = os.path.join(FC_dict, subject_id)
            for d1 in os.listdir(path):
                path1 = os.path.join(path, d1, 'average_features.npy')
                matrix = np.load(path1)
                FC_list.append(matrix)
        print(len(FC_list))
        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label



#########################ADHD



class fc_ADHD_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []


        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回



        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ADHDno(FC_dict)
        print(file_list)
        label_list = Reader.get_ADHD_score(file_list, score='DX')

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        # print(file_paths)
        sorted_paths = sorted(file_paths, key=lambda x: extract_number(x))
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            print(file_path)
            matrix = np.load(file_path)
            FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label




class stage2_ADHD_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):

        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sorted(file_paths, key=ADHD_sort_by_last_number)
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            print(file_path)
            matrix = np.load(file_path)
            FC_list.append(matrix)

        self.FCs = FC_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.FCs)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        return data


def sort_by_number(filename):
    return int(filename.split('.')[0])

class ADHD_ALL_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ADHD(FC_dict)
        print(file_list)
        label_list = Reader.get_subject_score_ADHD(file_list, score='DX')

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            file_paths_sorted = sorted(subject_list, key=sort_by_number)

            print(file_paths_sorted)
            for subject_id in file_paths_sorted:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label

class ADHD_ave_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ave(FC_dict)
        # print(file_list)
        label_list = Reader.get_subject_score_ave(file_list, score='DX')

        for subject_id in file_list:
            print(subject_id)
            path = os.path.join(FC_dict, subject_id)
            for d1 in os.listdir(path):
                # print(d1)
                path1 = os.path.join(path, d1, 'average_features.npy')
                matrix = np.load(path1)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label


class ADHD_std_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ave(FC_dict)
        # print(file_list)
        label_list = Reader.get_subject_score_ave(file_list, score='DX')

        for subject_id in file_list:
            print(subject_id)
            path = os.path.join(FC_dict, subject_id)
            for d1 in os.listdir(path):
                # print(d1)
                path1 = os.path.join(path, d1, 'std_features.npy')
                matrix = np.load(path1)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label



###abide2
class stage2_abide2_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):

        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sorted(file_paths, key=ADHD_sort_by_last_number)
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            print(file_path)
            matrix = np.load(file_path)
            FC_list.append(matrix)

        self.FCs = FC_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.FCs)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        return data

class abide2_ave_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        FC_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_ave(FC_dict)
        print(len(file_list))
        label_list = Reader.get_ABIDE2_subject_score_ave(file_list, score='DX_GROUP')
        print(len(label_list))
        for subject_id in file_list:
            # print(subject_id)
            path = os.path.join(FC_dict, subject_id)
            for d1 in os.listdir(path):
                path1 = os.path.join(path, d1, 'average_features.npy')
                matrix = np.load(path1)
                FC_list.append(matrix)
        print(len(FC_list))
        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label
class abide2_Dataset_official(Dataset):
    '''
    官网的数据版
    '''

    def __init__(self, FC_dict, save_dir=None):
        self.site_feature = {}
        FC_list = []
        label_list = []

        if save_dir:
            # 检查是否已经存在保存的数据文件
            if os.path.exists(os.path.join(save_dir, 'FCs.npy')) and os.path.exists(os.path.join(save_dir, 'labels.npy')):
                print("Loading dataset from saved files...")
                self.load_dataset(save_dir)
                return  # 直接加载数据后返回

        # 获取文件夹中所有.npy文件的列表
        file_list = Reader.get_ids_abide2(FC_dict)
        print(file_list)
        label_list = Reader.get_abide2_dfc(file_list, score='DX_GROUP')

        file_paths = glob.glob(os.path.join(FC_dict, '*'))
        sorted_paths = sort_paths_by_last_number(file_paths)
        print(sorted_paths)
        # 处理文件
        for file_path in sorted_paths:
            subject_list = os.listdir(file_path)
            # sorted_filenames = Reader.sort_filenames(subject_list)
            subject_list = [filename for filename in subject_list if re.search(r'\d+', filename)]
            file_paths_sorted = sorted(subject_list, key=sort_by_number)

            print(file_paths_sorted)

            for subject_id in file_paths_sorted:
                path = os.path.join(file_path, subject_id)
                matrix = np.load(path)
                FC_list.append(matrix)

        self.FCs = FC_list
        self.labels = label_list
        self.save_dir = save_dir

        # 如果提供了保存目录，保存数据
        if save_dir:
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'FCs.npy'), np.array(self.FCs))
        np.save(os.path.join(self.save_dir, 'labels.npy'), np.array(self.labels))

    def load_dataset(self, load_dir):
        self.FCs = np.load(os.path.join(load_dir, 'FCs.npy'), allow_pickle=True)
        self.labels = np.load(os.path.join(load_dir, 'labels.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.FCs[idx], dtype=torch.float32)
        data = data.unsqueeze(0)  # 添加通道数
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = F.one_hot(label.to(torch.int64), num_classes=2).float()
        return data, label