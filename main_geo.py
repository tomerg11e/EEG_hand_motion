from datetime import datetime
import EEGDataset
import utils
from termcolor import cprint
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
from Model import StaticGraphGNN
from pytorch_lightning.loggers import WandbLogger

"""
used for running a geometric learning task
"""

SEED = 28

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CHECKPOINT_PATH = "lstm_models"
project_name = "my-test-project"
entity = "tomerg11e"


def get_data_geo(data_path: str, arguments, upper_cap=None):
    """
    get geometric data from the given fila
    :param data_path: path where the data is located
    :param arguments: namespace containing batch size value and train val test ration
    :param upper_cap: takes only upper_cap samples from the data, useful for fast checking for errors
    :return:
    """
    data_transforms = Compose([EEGDataset.butter_filter])
    dataset = EEGDataset.EEGDataset(data_path.split('raw')[0], data_transform=data_transforms)
    # dataset = EEGDenseDataset(None, FULL_DATA_PATH, transform=None, upper_cap=upper_cap, data_transform=data_transforms)
    train, val, test = torch.utils.data.random_split(dataset,
                                                     [int(i * len(dataset)) for i in arguments.train_test_ratio])
    train_loader = DataLoader(train, batch_size=arguments.batch_size, shuffle=True, num_workers=arguments.num_workers)
    val_loader = DataLoader(val, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)
    test_loader = DataLoader(test, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers)
    return train_loader, val_loader, test_loader, dataset


def main(arguments):
    cprint(f"Configuration: {vars(arguments)}", color="green")

    data_path = "Data/raw/data_part0.npy"

    dl_train, dl_val, dl_test, dataset = get_data_geo(data_path, arguments, upper_cap=100)

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


if __name__ == '__main__':
    arguments_config = utils.get_configuration(upload=True)
    main(arguments_config)
