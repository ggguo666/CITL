import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
import os
from .read_abide_stats_parall import read_data
from .read_abide_stats_parall import read_data_stage3, read_data_stage1,read_data_stage_GCN_MDD

def extract_number(filename):
    parts = filename.split('_')
    main_part = parts[0]
    suffix = ''.join(filter(str.isdigit, parts[1].split('.')[0]))
    return (int(main_part), int(suffix))

class ABIDEDataset_stage3(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset_stage3, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    ## 加上@property，可以使得方法像属性一样被调用
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        # print(len(onlyfiles))
        # 按照文件名中的数字部分进行排序
        onlyfiles = sorted(onlyfiles, key=extract_number)
        print(len(onlyfiles))
        return onlyfiles
    #返回process()方法处理后的文件名列表
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        #是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch。
        self.data, self.slices = read_data_stage3(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))







class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    ## 加上@property，可以使得方法像属性一样被调用
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        # onlyfiles = sorted(onlyfiles, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))
        print(onlyfiles)
        return onlyfiles
    #返回process()方法处理后的文件名列表
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        #是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch。
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))



class ABIDEDataset_BaseModel(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset_BaseModel, self).__init__(root, transform, pre_transform)
        # print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    ## 加上@property，可以使得方法像属性一样被调用
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        # print(onlyfiles)
        return onlyfiles
    #返回process()方法处理后的文件名列表
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        #是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch。
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))




class ABIDEDataset_stage1(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset_stage1, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    ## 加上@property，可以使得方法像属性一样被调用
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles = sorted(onlyfiles, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))
        # print(onlyfiles)
        return onlyfiles

    # 返回process()方法处理后的文件名列表
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        # 是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch。
        self.data, self.slices = read_data_stage1(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

class ABIDEDataset_GCN_MDD(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset_GCN_MDD, self).__init__(root, transform, pre_transform)
        # print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    ## 加上@property，可以使得方法像属性一样被调用
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        # onlyfiles = sorted(onlyfiles, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))
        onlyfiles.sort()
        print(onlyfiles)
        return onlyfiles

    # 返回process()方法处理后的文件名列表
    @property
    def processed_file_names(self):
        return 'data.pt'
    def set_new_indices(self):
        self.__indices__ = list(range(self.len()))

    def process(self):
        # Read data into huge `Data` list.
        # 是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch。
        data_list = read_data_stage_GCN_MDD(self.raw_dir)
        print(data_list)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
