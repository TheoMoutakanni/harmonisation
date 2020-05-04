"""
Train an autonencoder
"""
import numpy as np

from harmonisation.datasets import SHDataset
from harmonisation.utils import (get_paths_ADNI, get_paths_PPMI,
                                 train_test_split)
from harmonisation.trainers import BaseTrainer
from harmonisation.models import ENet


from harmonisation.settings import SIGNAL_PARAMETERS

# Get the list of all ADNI files, splitted in train/test/validation
blacklist = ['003_S_4288_S142486']
paths, sites_dict = get_paths_PPMI()
path_train, path_validation, path_test = train_test_split(
    paths,
    test_proportion=0,
    validation_proportion=0.2,
    max_combined_size=10,
    blacklist=blacklist)

print("Train dataset size :", len(path_train))
print("Test dataset size :", len(path_test))
print("Validation dataset size", len(path_validation))

# Create two dataset, one for train and one for validation
train_dataset = SHDataset(path_train,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          signal_parameters=SIGNAL_PARAMETERS,
                          transformations=None,
                          normalize_data=True,
                          n_jobs=8,
                          cache_dir="./")

validation_dataset = SHDataset(path_validation,
                               patch_size=SIGNAL_PARAMETERS["patch_size"],
                               signal_parameters=SIGNAL_PARAMETERS,
                               transformations=None,
                               normalize_data=True,
                               mean=train_dataset.mean,
                               std=train_dataset.std,
                               n_jobs=8,
                               cache_dir="./")

# Create the newtork to train
net = ENet(patch_size=SIGNAL_PARAMETERS["patch_size"],
           sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=[32, 64, 128],
           encoder_relu=False,
           decoder_relu=True)

net = net.to("cuda")

save_folder = "./.saved_models/128-4-4-4/"

np.save(save_folder + 'mean_std.npy',
        [train_dataset.mean, train_dataset.std])

# Create the trainer
trainer = BaseTrainer(
    net,
    optimizer_parameters={
        "lr": 0.01,
        "weight_decay": 1e-8,
    },
    loss_specs={
        "type": "mse",
        "parameters": {}
    },
    metrics=["acc", "mse_gfa", "mse"],
    metric_to_maximize="mse",
    patience=100,
    save_folder=save_folder,
)

# Train the network
trainer.train(train_dataset,
              validation_dataset,
              num_epochs=11,
              batch_size=128,
              validation=True,
              freq_print=10,)
