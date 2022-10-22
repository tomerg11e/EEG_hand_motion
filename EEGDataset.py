import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset as geo_Dataset, Data
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

import os.path as osp
from util_def import Defaults as d
import torch
import scipy.signal as signal

data_format = "data_part"
labels_format = "labels_part"


class EEGDataset(geo_Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_transform=None):
        self.edge_index = None
        self.edge_weight = None
        self.x_all = None
        self.labels = None
        self.num_nodes = None
        self.data_transform = data_transform
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['data_part0.npy', 'data_part1.npy', 'data_part2.npy']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt']

    def download(self):
        pass

    def process(self):
        # create global vars
        full_data = []
        full_labels = []
        for raw_data in self.raw_paths:
            full_data.append(np.load(raw_data))
            full_labels.append(np.load(raw_data.replace(data_format, labels_format)))
        full_data = np.concatenate(full_data)
        full_data = full_data.reshape((full_data.shape[0], -1, full_data.shape[-1])).astype('float')
        self.num_nodes = full_data.shape[1]
        full_labels = np.concatenate(full_labels)
        if self.data_transform:
            full_data = self.data_transform(full_data)
        adjacency_matrix, degree_matrix, laplacian = corr(full_data)
        edges = np.stack(np.where(np.ones((self.num_nodes, self.num_nodes)) == 1))
        self.edge_index = torch.tensor(edges)
        self.edge_weight = torch.tensor(adjacency_matrix).flatten().float()
        self.x_all = torch.tensor(full_data).float()
        self.labels = full_labels

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x = self.x_all[idx]
        data = Data(x=x, edge_attr=self.edge_weight, edge_index=self.edge_index, y=self.labels[idx])
        return data


class EEGDenseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, upper_cap=None, data_transform=None):
        data = np.load(root_dir)
        data = data.reshape((data.shape[0], -1, data.shape[-1]))
        data = data[:upper_cap]
        assert len(data.shape) == 3
        self.transform = transform
        self.data_transform = data_transform
        if self.data_transform:
            data = self.data_transform(data)

        self.length, self.num_nodes, self.seq_len = data.shape
        self.labels = torch.tensor(np.load(root_dir.replace(data_format, labels_format))).long()[:self.length]
        adjacency_matrix, degree_matrix, laplacian = corr(data)
        edges = np.stack(np.where(np.ones((self.num_nodes, self.num_nodes)) == 1))
        self.edge_index = torch.tensor(edges)
        self.edge_weight = torch.tensor(adjacency_matrix)
        self.x_all = torch.tensor(data).float()

    def __getitem__(self, idx):
        sample = {'x': self.x_all[idx], 'y': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.length


class SampleButterFilter:
    def __init__(self, low_band: float = 0.5, high_band: float = 30, N: int = 5):
        assert low_band < high_band
        self.b, self.a, *_ = signal.butter(N, [low_band, high_band], fs=d.freq, btype='band')

    def __call__(self, sample):
        x = signal.lfilter(self.b, self.a, sample['x'], axis=1)

        return {'x': x, 'y': sample['y']}


def butter_filter(data, low_band: float = 0.5, high_band: float = 30, N: int = 5):
    b, a, *_ = signal.butter(N, [low_band, high_band], fs=d.freq, btype='band')
    return signal.lfilter(b, a, data, axis=1)


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
