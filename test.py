import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import seaborn as sns

FULL_DATA_PATH = "Data/raw/full_data.npy"
FULL_LABELS_PATH = "Data/raw/full_labels.npy"
DATA_CLASS_PATH = "Data/raw/data_c$.npy"
WANTED_SHAPE = (5, 7)


def main():
    separate_data_by_class()
    for i in [0,1,2]:
        corr_class = f"{i}"
        data = np.load(DATA_CLASS_PATH.replace("$", corr_class))
        data = data.reshape((data.shape[0], -1, data.shape[-1])).astype('float')
        adjacency_matrix = calc_corr(data)
        show_corr_matrix(adjacency_matrix, corr_class)


def show_full_matrix():
    matrix_path = "adjacency_matrix.npy"
    adjacency_matrix = None
    if path.exists(matrix_path):
        adjacency_matrix = np.load(matrix_path)
    else:
        data = np.load(FULL_DATA_PATH)
        data = data.reshape((data.shape[0], -1, data.shape[-1])).astype('float')
        adjacency_matrix = calc_corr(data)
        np.save(matrix_path, adjacency_matrix)
    show_corr_matrix(adjacency_matrix)


def show_corr_matrix(adjacency_matrix, corr_class=None, lower_per: int = 75):
    lower_bound = np.percentile(adjacency_matrix.flatten(), lower_per)
    class_str = ""
    if corr_class is not None:
        class_str = " for class " + corr_class

    #heatmap
    sns.heatmap(adjacency_matrix)
    plt.title("Heatmap of Cross Correlation" + class_str)
    plt.show()

    #histogram
    plt.hist(adjacency_matrix.flatten(), 50, facecolor='g', alpha=0.75, label="Cross Correlation")
    plt.axvline(x=lower_bound, color='r', linestyle='dashed', linewidth=2, label=f"the {lower_per}th percentile")
    plt.title("Histogram of Cross Correlation" + class_str)
    plt.legend()
    plt.xlim([0, 1])
    plt.show()

    #cross correlation graph
    adjacency_matrix = np.where(adjacency_matrix >= lower_bound, adjacency_matrix, 0)
    G = nx.from_numpy_matrix(np.matrix(adjacency_matrix), create_using=nx.DiGraph)
    pos = {i: (10 * (i % WANTED_SHAPE[1]), 10 * (WANTED_SHAPE[0] * WANTED_SHAPE[1] - i // WANTED_SHAPE[1])) for i in
           range(WANTED_SHAPE[0] * WANTED_SHAPE[1])}
    am = adjacency_matrix.flatten()
    am = np.log(am+1)
    am = (am - am.min()) / (am.max() - am.min())
    cmap = plt.cm.get_cmap("inferno")
    am = [cmap(val) for val in list(am)]
    plt.title(f"Graph of ({lower_per}th percentile) Cross Correlation" + class_str)
    nx.draw(G, pos, with_labels=True, edge_color=am)
    plt.show()


def calc_corr(data: np.ndarray):
    samples, channels, seq_len = data.shape

    res_data = np.transpose(data, axes=[1, 2, 0]).reshape(35, -1)
    pcc = np.corrcoef(res_data)
    pcc = pcc[:channels, :channels]
    appc = np.abs(pcc)

    adjacency_matrix = appc - np.eye(channels)

    return adjacency_matrix


def separate_data_by_class():
    data = np.load(FULL_DATA_PATH)
    data = data.reshape((data.shape[0], -1, data.shape[-1])).astype('float')
    labels = np.load(FULL_LABELS_PATH)
    for i in range(3):
        label_mask = (labels == i)
        relevant_data = data[label_mask]
        np.save(DATA_CLASS_PATH.replace("$", str(i)), relevant_data)
    print("a")


if __name__ == "__main__":
    main()
