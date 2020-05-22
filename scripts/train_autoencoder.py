"""
Train an autonencoder
"""
import numpy as np

from harmonisation.datasets import SHDataset
from harmonisation.utils import (get_paths_ADNI, get_paths_PPMI,
                                 train_test_split)
from harmonisation.trainers import BaseTrainer
from harmonisation.functions.shm import get_B_matrix
from harmonisation.models import ENet, UNet, DisNet


from harmonisation.settings import SIGNAL_PARAMETERS

# Get the list of all ADNI files, splitted in train/test/validation
blacklist = ['003_S_4288_S142486', '3169_BL_01',
             '3169_V04_00', '3168_BL_00', '3167_SC_01']
paths, sites_dict = get_paths_PPMI()
# paths = get_paths_ADNI()
path_train, path_validation, path_test = train_test_split(
    paths,
    test_proportion=0,
    validation_proportion=0.1,
    max_combined_size=10,
    blacklist=blacklist,
    seed=42)

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

sh_order = SIGNAL_PARAMETERS['sh_order']
ncoef = int((sh_order + 2) * (sh_order + 1) / 2)

B, invB = get_B_matrix(train_dataset.data[0]['gtab'], sh_order)

save_folder = "./.saved_models/bridge-IN-meanstd-dwimse/"

# Create the newtork to train
# net = ENet(sh_order=SIGNAL_PARAMETERS['sh_order'],
#            embed=[64, 128, 256],
#            encoder_relu=False,
#            decoder_relu=True)

net, _ = ENet.load(save_folder + '40_net.tar.gz')

# net = UNet(ncoef, ncoef, 16)
# net = DisNet(ncoef, ncoef, 16, 5, 256)

net = net.to("cuda")

np.save(save_folder + 'mean_std.npy',
        [train_dataset.mean, train_dataset.std])

# Create the trainer
trainer = BaseTrainer(
    net,
    optimizer_parameters={
        "lr": 0.01,
        "weight_decay": 1e-8,
    },
    loss_specs=[
        {"type": "mse",
         "parameters": {},
         "coeff": 1.},
        {"type": "dwi_mse",
         "parameters": {'B': B, 'voxels_to_take': 'center'},
         "coeff": 1.},
    ],
    metrics=["acc", "mse"],
    metric_to_maximize="mse",
    patience=100,
    save_folder=save_folder,
)

# Train the network
trainer.train(train_dataset,
              validation_dataset,
              num_epochs=41,
              batch_size=200,
              validation=True,
              freq_print=20,)
