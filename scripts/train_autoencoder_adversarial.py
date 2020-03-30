"""
Train an autoencoder with an adversarial loss
"""

from harmonisation.datasets import SHDataset
from harmonisation.utils import get_paths_ADNI, train_test_split
from harmonisation.trainers import AdversarialTrainer
from harmonisation.models import ENet, AdversarialNet


from harmonisation.settings import SIGNAL_PARAMETERS, DATASET_PARAMETERS

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
                          n_jobs=4,
                          cache_dir="./")

test_dataset = SHDataset(path_validation,
                         patch_size=SIGNAL_PARAMETERS["patch_size"],
                         signal_parameters=SIGNAL_PARAMETERS,
                         transformations=None,
                         mean=train_dataset.mean,
                         std=train_dataset.std,
                         n_jobs=4,
                         cache_dir="./")

net = ENet(patch_size=SIGNAL_PARAMETERS["patch_size"],
           sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=128,
           encoder_relu=False,
           decoder_relu=True)

adv_net = AdversarialNet(patch_size=SIGNAL_PARAMETERS["patch_size"],
                         sh_order=SIGNAL_PARAMETERS['sh_order'],
                         embed=128,
                         encoder_relu=True)

net = net.to("cuda")
adv_net = adv_net.to("cuda")


adv_trainer = AdversarialTrainer(
    net,
    adv_net,
    **DATASET_PARAMETERS
)

# Train both networks each epoch

adv_trainer.train_both(train_dataset,
                       test_dataset,
                       num_epochs=501,
                       batch_size=128,
                       validation=True)


# Train multiple epoch each network separately


adv_trainer.train_adv_net(train_dataset,
                          test_dataset,
                          num_epochs=20,
                          batch_size=128,
                          validation=True)


adv_trainer.train(train_dataset,
                  test_dataset,
                  num_epochs=251,
                  batch_size=128,
                  validation=True)

adv_trainer.train_adv_net(train_dataset,
                          test_dataset,
                          num_epochs=100,
                          batch_size=128,
                          validation=True)

adv_trainer.train(train_dataset,
                  test_dataset,
                  num_epochs=251,
                  batch_size=128,
                  validation=True)

adv_trainer.train_adv_net(train_dataset,
                          test_dataset,
                          num_epochs=100,
                          batch_size=128,
                          validation=True)

adv_trainer.train(train_dataset,
                  test_dataset,
                  num_epochs=501,
                  batch_size=128,
                  validation=True)
