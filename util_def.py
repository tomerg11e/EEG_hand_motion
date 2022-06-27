from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class Defaults:
    # training
    lr = 1e-3
    num_workers = 4
    batch_size = 32

    # model
    max_epochs = 20
    # max_epochs = 2
    hidden_size = 64
    num_layers = 2

    # Data specific
    freq = 500
    in_feature = 35
    seq_len = 450
    num_classes = 3
