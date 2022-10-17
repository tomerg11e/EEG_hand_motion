import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch.nn.functional import log_softmax, softmax
import torch_geometric.nn as geom_nn
from typing import Optional
import torch_geometric.data as geom_data
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool, SAGPooling, max_pool
import math

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class LstmModel(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, num_layers: int, seq_length: int, num_classes: int = 3):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=in_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def create_zero_hidden(self, x):
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        return h_0, c_0

    def forward(self, x, hidden):
        if hidden is None:
            hidden = self.create_zero_hidden(x)

        # out, (h, c) = self.lstm(x, hidden)
        out, (h, c) = self.lstm(x)
        out = self.head(torch.tanh(h[-1]))
        out = log_softmax(out, dim=1)
        return out, h


class GNNModel(nn.Module):

    def __init__(self, in_feature, hidden_size, num_classes, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            in_feature - Dimension of input features
            hidden_size - Dimension of hidden features
            num_classes - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = in_feature, hidden_size
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = hidden_size
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=num_classes,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index, edge_weight)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, in_feature, hidden_size, num_classes, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            in_feature - Dimension of input features
            hidden_size - Dimension of hidden features
            num_classes - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(in_feature=in_feature,
                            hidden_size=hidden_size,
                            num_classes=hidden_size,  # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, edge_index, edge_weight, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index, edge_weight)
        x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling
        x = self.head(x)
        return x


class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, edge_weight, batch_idx = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.model(x, edge_index, edge_weight, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.num_classes == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


class DenseStaticGraphGNN(pl.LightningModule):

    def __init__(self, dataset, **kwargs):
        super(DenseStaticGraphGNN, self).__init__()
        self.edge_index = dataset.edge_index
        self.edge_weight = dataset.edge_weight.reshape(-1, 1).float()
        self.in_feature = kwargs["in_feature"]
        self.num_classes = kwargs["num_classes"]
        self.save_hyperparameters()

        # hidden layer node features
        self.hidden = kwargs["hidden_size"]
        self.seq_len = kwargs["seq_len"]
        self.model = Sequential("x, edge_index, edge_weight", [
            (GCNConv(self.seq_len, 128, node_dim=1), "x, edge_index, edge_weight -> x1"),
            (GCNConv(128, 64, node_dim=1), "x1, edge_index, edge_weight -> x2"),
            (ReLU(), "x2 -> x2a"), (Dropout(p=0.2), "x2a -> x2d"),
            (GCNConv(64, 32, node_dim=1), "x2d, edge_index, edge_weight -> x3"),
            (GCNConv(32, 16, node_dim=1), "x3, edge_index, edge_weight -> x4")])

        # self.conv1 = GCNConv(self.seq_len, 128, node_dim=1)
        # self.pool1 = SAGPooling(in_channels=128, node_dim=1)
        # self.conv2 = GCNConv(128, 64, node_dim=1)
        # self.pool2 = SAGPooling(in_channels=64)
        # self.conv3 = GCNConv(64, 32, node_dim=1)
        # self.pool3 = SAGPooling(in_channels=32)
        # self.conv4 = GCNConv(32, 16, node_dim=1)

        # self.model = Sequential("x, edge_index, edge_weight", [
        #     (GCNConv(self.seq_len, 16, node_dim=1), "x, edge_index, edge_weight -> x1")])

    def forward(self, x, batch_index):
        # x_out = self.model(x, self.edge_index, self.edge_weight)
        # x_out = global_mean_pool(x_out.transpose(1, 0), None).transpose(1, 0)
        # x_out = Linear(16, self.num_classes)(x_out).squeeze()
        # x_out = torch.softmax(x_out, dim=1)
        x_out = self.conv1(x, self.edge_index, self.edge_weight)
        x_out = self.pool1(x_out, self.edge_index, self.edge_weight)
        return x_out

    # c = torch.tensor([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7])

    def training_step(self, batch, batch_index):
        x = batch["x"]
        y = batch["y"]

        x_out = self.forward(x, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)
        accuracy = (pred == y).sum() / pred.shape[0]

        self.log("loss/train1", loss, on_epoch=True, on_step=False)
        self.log("acc/train1", accuracy, on_epoch=True, on_step=False)

        return {"loss": loss, "accuracy": accuracy, "size": len(y)}

    def training_epoch_end(self, training_step_outputs):
        train_loss = 0.0
        num_correct = 0
        num_total = 0

        for output in training_step_outputs:
            train_loss += output["loss"]

            num_correct += output["accuracy"] * output["size"]
            num_total += output["size"]

        train_accuracy = num_correct / num_total
        train_loss = train_loss / num_total

        # self.log("acc/train1", train_accuracy)
        # self.log("loss/train1", train_loss)

    def validation_step(self, batch, batch_index):
        x = batch["x"]
        y = batch["y"]

        x_out = self.forward(x, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)

        # self.log("loss/train", loss, on_epoch=True, prog_bar=True)
        # self.log("acc/train", accuracy, on_epoch=True, prog_bar=True)

        return x_out, pred, y

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in validation_step_outputs:
            val_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]

        val_accuracy = num_correct / num_total
        val_loss = val_loss / num_total

        self.log("acc/val", val_accuracy)
        self.log("loss/val", val_loss)

    def test_step(self, batch, batch_index):
        x = batch["x"]
        y = batch["y"]

        x_out = self.forward(x, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)

        return x_out, pred, y

    def test_epoch_end(self, test_step_outputs):
        test_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in test_step_outputs:
            test_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]

        test_accuracy = num_correct / num_total
        test_loss = test_loss / num_total

        self.log("acc/test", test_accuracy)
        self.log("loss/test", test_loss)

        return {'test_loss': test_loss, 'test_acc': test_accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


class GNNStep(nn.Module):
    def __init__(self, cin_feature, cout_feature, lin_feature, lout_feature):
        super().__init__()
        self.cin_feature = cin_feature
        self.cout_feature = cout_feature
        self.lin_feature = lin_feature
        self.lout_feature = lout_feature

        self.conv = GCNConv(cin_feature, cout_feature)
        self.bn = geom_nn.BatchNorm(cin_feature)
        self.activation = ReLU()
        self.dropout = Dropout(p=0.5)
        self.pool = SAGPooling(cout_feature)
        self.linear = Linear(lin_feature, lout_feature)

    def forward(self, x, edge_index, edge_attr, _batch):
        x1 = self.conv(self.bn(x), edge_index, edge_attr)
        x1 = self.dropout(self.activation(x1))
        x1, edge_index, edge_attr, _batch1, *_ = self.pool(x1, edge_index, edge_attr, _batch)

        unique = torch.unique(_batch1)
        x1l = torch.dstack([x1[_batch1 == i] for i in unique]).permute(2, 0, 1)

        return x1, edge_index, edge_attr, _batch1, self.linear(x1l.flatten(start_dim=1))


class StaticGraphGNN(pl.LightningModule):

    def __init__(self, **kwargs):
        super(StaticGraphGNN, self).__init__()
        self.in_feature = kwargs["in_feature"]
        self.num_classes = kwargs["num_classes"]
        self.hidden_size = kwargs["hidden_size"]
        self.seq_len = kwargs["seq_len"]
        self.dims = [self.seq_len, 128, 64]
        self.num_layers = len(self.dims) - 1
        for i in range(self.num_layers):
            self.add_module(f"GNNStep_{i}", GNNStep(self.dims[i], self.dims[i + 1],
                                                    self.dims[i + 1] * math.ceil(self.in_feature / (2 ** (i + 1))),
                                                    self.hidden_size))

        self.linear = Linear(self.hidden_size * self.num_layers, self.num_classes)
        self.dropout = Dropout(p=0.5)
        self.save_hyperparameters()
        self.lr = kwargs["learning_rate"]

    def forward(self, batch, batch_index):
        x, edge_index, edge_attr, _batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        multi_level = []
        for i in range(self.num_layers):
            module = self.get_submodule(f"GNNStep_{i}")
            x, edge_index, edge_attr, _batch, xl = module(x, edge_index, edge_attr, _batch)
            multi_level.append(F.normalize(xl, dim=1))

        multi_level = torch.hstack(multi_level)
        multi_level = self.dropout(multi_level)
        output = self.linear(multi_level)
        return softmax(output, dim=1)

    def training_step(self, batch, batch_index):
        y = torch.tensor(batch.y).long()
        x_out = self.forward(batch, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)
        accuracy = (pred == y).sum() / pred.shape[0]

        self.log("loss/train", loss, on_epoch=True, on_step=False)
        self.log("acc/train", accuracy, on_epoch=True, on_step=False)

        return {"loss": loss, "accuracy": accuracy, "size": len(y)}

    # def training_epoch_end(self, training_step_outputs):
    #     train_loss = 0.0
    #     num_correct = 0
    #     num_total = 0
    #
    #     for output in training_step_outputs:
    #         train_loss += output["loss"]
    #
    #         num_correct += output["accuracy"] * output["size"]
    #         num_total += output["size"]
    #
    #     train_accuracy = num_correct / num_total
    #     train_loss = train_loss / num_total
    #
    #     # self.log("acc/train1", train_accuracy)
    #     # self.log("loss/train1", train_loss)

    def validation_step(self, batch, batch_index):
        y = torch.tensor(batch.y).long()
        x_out = self.forward(batch, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)

        return x_out, pred, y

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in validation_step_outputs:
            val_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]

        val_accuracy = num_correct / num_total
        val_loss = val_loss / num_total

        self.log("acc/val", val_accuracy)
        self.log("loss/val", val_loss)

    def test_step(self, batch, batch_index):
        y = torch.tensor(batch.y).long()
        x_out = self.forward(batch, batch_index)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)

        return x_out, pred, y

    def test_epoch_end(self, test_step_outputs):
        test_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in test_step_outputs:
            test_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]

        test_accuracy = num_correct / num_total
        test_loss = test_loss / num_total

        self.log("acc/test", test_accuracy)
        self.log("loss/test", test_loss)

        return {'test_loss': test_loss, 'test_acc': test_accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
