from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import EEGDataset
import utils
from termcolor import cprint
import torch
from tqdm import tqdm
import torch.utils.data as data
import pytorch_lightning as pl
from scipy import sparse
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import torchvision.transforms as vision_transforms
from torchvision.transforms import Compose
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from EEGDataset import EEGDenseDataset, butter_filter
from Model import GraphLevelGNN, StaticGraphGNN
from pytorch_lightning.loggers import WandbLogger
from util_def import Defaults as d
import wandb

SEED = 28

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CHECKPOINT_PATH = "lstm_models"
project_name = "my-test-project"
entity = "tomerg11e"


def get_data_geo(data_path: str, train_ration: tuple, arguments, upper_cap=None):
    data_transforms = Compose([
        butter_filter
    ])
    dataset = EEGDataset.EEGDataset(data_path.split('raw')[0], data_transform=data_transforms)
    # dataset = EEGDenseDataset(None, FULL_DATA_PATH, transform=None, upper_cap=upper_cap, data_transform=data_transforms)
    train, val, test = torch.utils.data.random_split(dataset, [int(i * len(dataset)) for i in train_ration])
    # data = np.load(data_path).astype('float64')
    # data = data.reshape((data.shape[0], -1, data.shape[-1]))
    # labels = np.load(data_path.replace(data_format, labels_format))
    # adjacency_matrix, degree_matrix, laplacian = corr(data)
    # train = []
    # val = []
    # test = []
    # num_nodes = arguments.in_feature
    # num_nodes = 35
    # edges = np.stack(np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes) == 1))
    # edges = np.stack(np.where(np.ones((num_nodes, num_nodes)) == 1))
    # edges = torch.tensor(edges, dtype=torch.long)
    # edge_attr = adjacency_matrix.reshape(-1, 1)
    # sample_graph = None
    # for sample, label in zip(data, labels):
    #     r = np.random.rand(1)
    #     sample_graph = Data(x=torch.tensor(sample), edge_index=edges, edge_attr=torch.tensor(edge_attr),
    #                         y=label.reshape(1, 1), num_nodes=num_nodes)
    #     if r < train_ration[0]:
    #         train.append(sample_graph)
    #     elif r < train_ration[1]:
    #         train.append(sample_graph)
    #     else:
    #         test.append(sample_graph)
    # train_loader = DataLoader(train, batch_size=arguments.batch_size, shuffle=True, num_workers=arguments.num_workers)
    # val_loader = DataLoader(val, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)
    # test_loader = DataLoader(test, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)

    train_loader = DataLoader(train, batch_size=arguments.batch_size, shuffle=True, num_workers=arguments.num_workers)
    val_loader = DataLoader(val, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)
    test_loader = DataLoader(test, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)
    return train_loader, val_loader, test_loader, dataset


def main(arguments):
    cprint(f"Configuration: {vars(arguments)}", color="green")

    data_path = "Data/raw/data_part0.npy"

    # <editor-fold desc="HYPER-PARAMETERS">
    learning_rate = arguments.learning_rate
    max_epochs = arguments.max_epochs

    # model_type = arguments.model_type
    batch_size = arguments.batch_size
    num_workers = arguments.num_workers
    train_test_ratio = arguments.train_test_ratio

    # data params
    in_feature = arguments.in_feature  # from data
    # seq_len = arguments.seq_len  # from data
    num_classes = arguments.num_classes  # from data
    # freq = arguments.freq  # from data

    # model params
    hidden_size = arguments.hidden_size
    num_layers = arguments.num_layers
    # </editor-fold>

    dl_train, dl_val, dl_test, dataset = get_data_geo(data_path, train_test_ratio, arguments, upper_cap=100)

    model = StaticGraphGNN(dataset=dataset, **vars(arguments))
    train_lightning_model("test", model, dl_train, dl_val, dl_test, **vars(arguments))


def train_lightning_model(model_name, model, train_loader, val_loader, test_loader, max_epochs, wandb_upload, **kwargs):
    pl.seed_everything(SEED)
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(MODEL_CHECKPOINT_PATH, "miniModel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    logger = True
    if wandb_upload:
        date_time = datetime.now().strftime("%d_%m_%y__%H_%M_%S")
        run_name = f"miniModel, {date_time}"
        logger = WandbLogger(name=run_name, project=project_name, entity=entity, config={**kwargs})
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="acc/val")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=max_epochs,
                         log_every_n_steps=5,
                         logger=logger)

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    # train_result = trainer.test(model, test_dataloaders=train_loader, verbose=True)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    result = {"test": test_result[0]['acc/test']}
    return model, result


def corr(data, plot: bool = False):
    samples, channels, seq_len = data.shape

    # do corrcoef to all
    res_data = np.transpose(data, axes=[1, 2, 0]).reshape(35, -1)
    pcc = np.corrcoef(res_data)

    pcc = pcc[:channels, :channels]
    appc = np.abs(pcc)
    adjacency_matrix = appc - np.eye(channels)
    degree_matrix = np.sum(adjacency_matrix, axis=0)
    laplacian = np.diag(degree_matrix) - adjacency_matrix

    if plot:
        sns.heatmap(pcc)
        plt.title("PCC")
        plt.show()
        sns.heatmap(appc)
        plt.title("abs PCC matrix")
        plt.show()
        sns.heatmap(adjacency_matrix)
        plt.title("Adjacency matrix")
        plt.show()
        sns.heatmap(laplacian)
        plt.title("laplacian matrix")
        plt.show()

    return adjacency_matrix, degree_matrix, laplacian


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    # calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step


def train_graph_classifier(model_name, train_loader, val_loader, test_loader, **model_kwargs):
    pl.seed_everything(SEED)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(MODEL_CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=500,
                         progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(MODEL_CHECKPOINT_PATH, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(SEED)
        model = GraphLevelGNN(**model_kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    return model, result


if __name__ == '__main__':
    arguments_config = utils.get_configuration(upload=True)
    main(arguments_config)

    # corr()
