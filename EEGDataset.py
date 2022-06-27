import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset as geo_Dataset, Data
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

import os.path as osp

import torch

data_format = "data_part"
labels_format = "labels_part"


class EEGDataset(geo_Dataset):

    def EEGDataset(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['data_part0.npy', 'data_part1.npy']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt']

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_data in self.raw_paths:
            data = np.load(raw_data)
            data = data.reshape((data.shape[0], -1, data.shape[-1]))
            labels = np.load(raw_data.replace(EEGDataset.data_format, EEGDataset.labels_format))
            adjacency_matrix, degree_matrix, laplacian = corr(data)
            """
            x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes."""
            sample = Data(x=data[0], edge_weight=adjacency_matrix, y=labels[0])

            if self.pre_filter is not None and not self.pre_filter(sample):
                continue

            if self.pre_transform is not None:
                sample = self.pre_transform(sample)

            torch.save(sample, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


class EEGDenseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, upper_cap=None):
        data = np.load(root_dir)
        data = data.reshape((data.shape[0], -1, data.shape[-1]))
        data = data[:upper_cap]
        assert len(data.shape) == 3
        self.length, self.num_nodes, self.seq_len = data.shape
        self.labels = torch.tensor(np.load(root_dir.replace(data_format, labels_format))).long()[:self.length]
        adjacency_matrix, degree_matrix, laplacian = corr(data)
        edges = np.stack(np.where(np.ones((self.num_nodes, self.num_nodes)) == 1))
        self.edge_index = torch.tensor(edges)
        self.edge_weight = torch.tensor(adjacency_matrix)
        self.x_all = torch.tensor(data).float()

    def __getitem__(self, idx):
        return {'x': self.x_all[idx], 'y': self.labels[idx]}

    def __len__(self):
        return self.length


def corr(data: np.ndarray):
    samples, channels, seq_len = data.shape
    res_data = np.transpose(data, axes=[1, 2, 0]).reshape(35, -1)
    pcc = np.corrcoef(res_data)
    pcc = pcc[:channels, :channels]
    appc = np.abs(pcc)

    adjacency_matrix = appc - np.eye(channels)
    degree_matrix = np.sum(adjacency_matrix, axis=0)
    laplacian = np.diag(degree_matrix) - adjacency_matrix

    return adjacency_matrix, degree_matrix, laplacian
