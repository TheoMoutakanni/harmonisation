"""
Train an autonencoder
"""

from harmonisation.datasets import SHDataset
from harmonisation.utils import get_paths_ADNI, train_test_split
from harmonisation.trainers import BaseTrainer
from harmonisation.models import ENet


from harmonisation.settings import SIGNAL_PARAMETERS

blacklist = []
paths = get_paths_ADNI()
path_train, path_validation, path_test = train_test_split(
    paths,
    test_proportion=0,
    validation_proportion=0.2,
    max_combined_size=10,
    blacklist=blacklist)

print("Train dataset size :", len(path_train))
print("Test dataset size :", len(path_test))
print("Validation dataset size", len(path_validation))


train_dataset = SHDataset(path_train,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          signal_parameters=SIGNAL_PARAMETERS,
                          transformations=None,
                          normalize_data=True,
                          n_jobs=8,
                          cache_dir="./")

test_dataset = SHDataset(path_validation,
                         patch_size=SIGNAL_PARAMETERS["patch_size"],
                         signal_parameters=SIGNAL_PARAMETERS,
                         transformations=None,
                         normalize_data=True,
                         mean=train_dataset.mean,
                         std=train_dataset.std,
                         n_jobs=8,
                         cache_dir="./")

net = ENet(patch_size=SIGNAL_PARAMETERS["patch_size"],
           sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=128,
           encoder_relu=False,
           decoder_relu=True)

net = net.to("cuda")


trainer = BaseTrainer(
    net,
    optimizer_parameters={
        "lr": 0.01,
        "weight_decay": 1e-8,
    },
    loss_specs={
        "type": "acc_mse",
        "parameters": {}
    },
    metrics=["acc", "mse"],
    metric_to_maximize="acc",
    patience=100,
    save_folder=None,
)

trainer.train(train_dataset,
              test_dataset,
              num_epochs=101,
              batch_size=16,
              validation=True)
