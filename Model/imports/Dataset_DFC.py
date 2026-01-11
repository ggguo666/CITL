from pathlib import Path
from imports import preprocess_data as Reader
# import preprocess_data as Reader
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import os
from os import listdir
import os.path as osp
# import utils
from imports import utils
import deepdish as dd
class ConnectivityData(InMemoryDataset):
    """ Dataset for the connectivity data."""

    def __init__(self,
                 root):
        super(ConnectivityData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob("*.npy")))
        return [str(file_path.name) for file_path in file_paths]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def set_new_indices(self):
        self.__indices__ = list(range(self.len()))

    def process(self):

        onlyfiles = [f for f in listdir(self.raw_dir) if osp.isfile(osp.join(self.raw_dir, f))]
        # onlyfiles = sorted(onlyfiles, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))
        onlyfiles.sort()
        print(onlyfiles)

        data_list = []

        for file_name in onlyfiles:
            temp = dd.io.load(os.path.join(self.raw_dir, file_name))
            print(file_name)

            y = temp['label'][()]
            y = int(y[0])
            print(y)
            y = torch.tensor([y]).long()
            connectivity = temp['corr'][0][()]
            np.fill_diagonal(connectivity, 0)
            x = torch.from_numpy(connectivity).float()
            # print(x.shape)
            adj = utils.compute_KNN_graph(connectivity)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))


        print(len(data_list))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

# if __name__ == '__main__':
#     dataset = ConnectivityData('/home/user/data/gsj/abide160/2222/stage1correlation/')