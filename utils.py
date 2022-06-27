import argparse
import wandb
import util_def
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
from util_def import Defaults as d


def get_configuration(upload: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", default=d.lr, type=float)
    # parser.add_argument("--regularization", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=d.max_epochs, type=int)
    parser.add_argument("--train_test_ratio", default=(0.8, 0.1, 0.1), type=tuple)

    parser.add_argument("--model_type", choices=["LSTM"], default="LSTM")
    parser.add_argument("--batch_size", default=d.batch_size, type=int)  # The batch size for the data loaders
    parser.add_argument("--num_workers", default=d.num_workers, type=int)  # The number of workers for the data loaders
    parser.add_argument("--hidden_size", default=d.hidden_size, type=int)
    parser.add_argument("--num_layers", default=d.num_layers, type=int)
    arguments = parser.parse_args()

    debugging = sys.gettrace() is not None
    arguments.debugging = debugging
    arguments.wandb_upload = not arguments.debugging
    if not upload:
        arguments.wandb_upload = upload

    arguments.in_feature = d.in_feature
    arguments.num_classes = d.num_classes
    arguments.seq_len = d.seq_len
    arguments.freq = d.freq

    return arguments


def upload_mean(epoch, train_result, test_result, verbose, wandb_upload):
    if wandb_upload:
        train_loss = sum(train_result.losses) / len(train_result.losses)
        test_loss = sum(test_result.losses) / len(test_result.losses)
        output_dict = {"train_loss": train_loss, "train_acc": train_result.accuracy,
                       "test_loss": test_loss, "test_acc": test_result.accuracy,
                       "Epoch": epoch}
        if verbose:
            print(output_dict)
        wandb.log(output_dict, step=epoch)


def plot_fit(fit_res: util_def.FitResult, fig=None, log_loss=False, legend=None, train_test_overlay: bool = False, ):
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
