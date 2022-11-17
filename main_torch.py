import numpy as np
import matplotlib.pyplot as plt
import os
from Model import LstmModel as Model
import torch
import torch.nn as nn
from training import RNNTrainer
import torch.utils.data
from datetime import datetime
import wandb
import argparse
from termcolor import cprint
import scipy.signal as signal
import utils
from util_def import Defaults as d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CHECKPOINT_PATH = "lstm_models"
project_name = "my-test-project"
entity = "tomerg11e"

"""
used for running a tabular learning task. not used in the final model
"""


def get_data_torch(data_path: str, labels_path: str, train_test_ratio: float = 0.8, batch_size: int = 32):
    """
    get the data from the paths

    :param data_path: the path in which the data are at in in .npy format
    :param labels_path: the path in which the labels are at is in in .npy format
    :param train_test_ratio: the ratio in which to split the data
    :param batch_size: the size of each batch
    :return: two dataloader object, train dataloader and test dataloader
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


def main(arguments):
    cprint(f"Configuration: {vars(arguments)}", color="green")

    data_path = "Data/data_part0.npy"
    labels_path = "Data/labels_part0.npy"

    # <editor-fold desc="HYPER-PARAMETERS">
    learning_rate = arguments.learning_rate
    # regularization = wandb.config["regularization"]
    num_epochs = arguments.num_epochs

    model_type = arguments.model_type
    batch_size = arguments.batch_size
    num_workers = arguments.num_workers
    train_test_ratio = arguments.train_test_ratio

    # model params
    in_dim = 35  # from data
    seq_len = 450  # from data
    hidden_dim = arguments.hidden_dim
    n_layers = arguments.n_layers
    # </editor-fold>

    dl_train, dl_test = get_data_torch(data_path, labels_path, train_test_ratio, batch_size)

    # model initialization
    loss_fn = nn.NLLLoss()
    model = Model(in_feature=in_dim, hidden_size=hidden_dim, num_layers=n_layers, seq_length=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = RNNTrainer(model, loss_fn, optimizer, device)

    # dir initialization
    os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)
    date_time = datetime.now().strftime("%d_%m_%y__%H_%M_%S")
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_PATH, date_time)
    os.mkdir(checkpoint_path)

    if arguments.wandb_upload:
        run_name = f"{model_type}, h={arguments.hidden_dim}, n_layers= {arguments.n_layers}"
        wandb.init(name=run_name, project=project_name, entity=entity,
                   config={**vars(arguments)}, reinit=True)

    # train the model
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, print_every=1, post_epoch_fn=utils.upload_mean,
                          checkpoints=os.path.join(checkpoint_path, "epoch_num"), wandb_upload=arguments.wandb_upload)

    # plotting and saving fig
    fig, axes = utils.plot_fit(fit_res)
    plt.savefig(os.path.join(checkpoint_path, "final_fig.png"))
    plt.show()


if __name__ == '__main__':
    arguments_config = utils.get_configuration()
    main(arguments_config)
