"""
Train an autoencoder with an adversarial loss
"""

from harmonisation.datasets import SHDataset
from harmonisation.utils import get_paths_PPMI, get_paths_ADNI, train_test_split
from harmonisation.trainers import AdversarialTrainer
from harmonisation.models import ENet, AdversarialNet


from harmonisation.settings import SIGNAL_PARAMETERS, TRAINER_PARAMETERS

blacklist = ['003_S_4288_S142486', '3169_BL_01',
             '3169_V04_00', '3168_BL_00', '3167_SC_01']
paths_ppmi, sites_dict = get_paths_PPMI()

for i in range(len(paths_ppmi)):
    paths_ppmi[i]['site'] = 0

path_train_ppmi, path_validation_ppmi, path_test_ppmi = train_test_split(
    paths_ppmi,
    test_proportion=0,
    validation_proportion=0.5,
    max_combined_size=10,
    blacklist=blacklist)

paths_adni = get_paths_ADNI()

for i in range(len(paths_adni)):
    paths_adni[i]['site'] = 1

path_train_adni, path_validation_adni, path_test_adni = train_test_split(
    paths_adni,
    test_proportion=0,
    validation_proportion=0.5,
    max_combined_size=10,
    blacklist=blacklist)

path_train = path_train_adni + path_train_ppmi
path_validation = path_validation_adni + path_validation_ppmi
path_test = path_test_adni + path_test_ppmi

print("Train dataset size :", len(path_train))
print("Test dataset size :", len(path_test))
print("Validation dataset size", len(path_validation))

SIGNAL_PARAMETERS['overlap_coeff'] = 1

train_dataset = SHDataset(path_train,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          signal_parameters=SIGNAL_PARAMETERS,
                          transformations=None,
                          normalize_data=True,
                          n_jobs=8,
                          cache_dir="./")

# SIGNAL_PARAMETERS['overlap_coeff'] = 2

validation_dataset = SHDataset(path_validation,
                               patch_size=SIGNAL_PARAMETERS["patch_size"],
                               signal_parameters=SIGNAL_PARAMETERS,
                               transformations=None,
                               normalize_data=True,
                               mean=train_dataset.mean,
                               std=train_dataset.std,
                               n_jobs=8,
                               cache_dir="./")

print('train_sites')
for d in train_dataset.data:
    print(d['site'])
print('val sites')
for d in validation_dataset.data:
    print(d['site'])

net = ENet(sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=[32, 64, 128],
           encoder_relu=False,
           decoder_relu=True)

# adv_net = AdversarialNet(patch_size=SIGNAL_PARAMETERS["patch_size"],
#                          sh_order=SIGNAL_PARAMETERS['sh_order'],
#                          embed=[16, 32, 64],
#                          number_of_classes=len(sites_dict),
#                          encoder_relu=True)

sh_order = SIGNAL_PARAMETERS['sh_order']
ncoef = int((sh_order + 2) * (sh_order + 1) / 2)

adv_net = AdversarialNet(in_dim=ncoef,
                         out_dim=len(sites_dict),
                         num_filters=4,
                         nb_layers=4,
                         embed_size=256)

net = net.to("cuda")
adv_net = adv_net.to("cuda")


adv_trainer = AdversarialTrainer(
    net,
    adv_net,
    **TRAINER_PARAMETERS
)

# Train both networks each epoch

adv_net, adv_metrics = adv_trainer.train_adv_net(train_dataset,
                                                 validation_dataset,
                                                 coeff_fake=0,
                                                 num_epochs=20,
                                                 batch_size=128,
                                                 validation=True)

adv_trainer.adv_net = adv_net

net, metrics = adv_trainer.train(train_dataset,
                                 validation_dataset,
                                 num_epochs=40,
                                 batch_size=128,
                                 validation=True)

adv_trainer.net = net

adv_trainer.train_both(train_dataset,
                       validation_dataset,
                       coeff_fake=1,
                       num_epochs=21,
                       batch_size=128,
                       validation=True)


# Train multiple epoch each network separately


# adv_net, adv_metrics = adv_trainer.train_adv_net(train_dataset,
#                                                  validation_dataset,
#                                                  num_epochs=20,
#                                                  batch_size=128,
#                                                  validation=True)

# adv_trainer.adv_net = adv_net

# net, metrics = adv_trainer.train(train_dataset,
#                                  validation_dataset,
#                                  num_epochs=40,
#                                  batch_size=128,
#                                  validation=True)

# adv_trainer.net = net

# adv_net, adv_metrics = adv_trainer.train_adv_net(train_dataset,
#                                                  validation_dataset,
#                                                  num_epochs=40,
#                                                  batch_size=128,
#                                                  validation=True)

# adv_trainer.adv_net = adv_net

# net, metrics = adv_trainer.train(train_dataset,
#                                  validation_dataset,
#                                  num_epochs=40,
#                                  batch_size=128,
#                                  validation=True,
#                                  metrics_final=metrics)

# adv_trainer.net = net

# adv_net, adv_metrics = adv_trainer.train_adv_net(train_dataset,
#                                                  validation_dataset,
#                                                  num_epochs=40,
#                                                  batch_size=128,
#                                                  validation=True)

# adv_trainer.adv_net = adv_net

# net, metrics = adv_trainer.train(train_dataset,
#                                  validation_dataset,
#                                  num_epochs=40,
#                                  batch_size=128,
#                                  validation=True,
#                                  metrics_final=metrics)
