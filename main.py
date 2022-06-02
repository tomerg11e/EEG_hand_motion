import itertools

import numpy as np
import matplotlib.pyplot as plt
import os
from Model import Model
import torch
import torch.nn as nn
from training import RNNTrainer
import torch.utils.data
from datetime import datetime

from typing_classes import FitResult

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CHECKPOINT_PATH = "lstm_models"


def main():
    # trainer params
    lr = 0.01
    num_epochs = 5

    # model params
    in_dim = 35  # from data
    seq_len = 450  # from data
    h_dim = 64
    n_layers = 1

    data_path = "Data/data_part0.npy"
    labels_path = "Data/labels_part0.npy"
    train_test_ratio = 0.8
    batch_size = 32
    dl_train, dl_test = get_data(data_path, labels_path, train_test_ratio, batch_size)

    loss_fn = nn.NLLLoss()
    model = Model(in_feature=in_dim, hidden_size=h_dim, num_layers=n_layers, seq_length=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = RNNTrainer(model, loss_fn, optimizer, device)

    os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)
    date_time = datetime.now().strftime("%d_%m_%y__%H_%M_%S")
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_PATH, date_time)
    os.mkdir(checkpoint_path)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=None, print_every=1,
                          checkpoints=os.path.join(checkpoint_path, "epoch_num"))
    fig, axes = plot_fit(fit_res)
    plt.savefig(os.path.join(checkpoint_path, "final_fig.png"))
    plt.show()


def get_data(data_path: str, labels_path: str, train_test_ratio: float = 0.8, batch_size: int = 32):
    """
    return the data inputted as path as train and test dataloader
    :param batch_size:
    :param train_test_ratio:
    :param data_path:
    :param labels_path:
    :return:
    """
    data = np.load(data_path)
    data = data.reshape((data.shape[0], -1, data.shape[-1]))
    labels = np.load(labels_path).reshape(-1, 1)
    data = torch.from_numpy(data).transpose(1, 2)  # (N, seq_length, in_features)
    labels = torch.from_numpy(labels)  # (N, 1)
    num_train = int(train_test_ratio * len(data))

    ds_train = torch.utils.data.TensorDataset(data[:num_train], labels[:num_train])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(data[num_train:], labels[num_train:])
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True)

    return dl_train, dl_test


def plot_fit(
        fit_res: FitResult,
        fig=None,
        log_loss=False,
        legend=None,
        train_test_overlay: bool = False,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "test"]), enumerate(["loss", "acc"]))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data = getattr(fit_res, attr)
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes


if __name__ == '__main__':
    main()
