"""
Train an autoencoder with an adversarial loss
"""

from harmonisation.datasets import SHDataset, AdversarialDataset
from harmonisation.utils import (get_paths_PPMI, get_paths_ADNI, get_paths_SIMON,
                                 train_test_val_split)
from harmonisation.trainers import StyleTrainer
from harmonisation.models import ENet, AdversarialNet
from harmonisation.functions.shm import get_B_matrix

from harmonisation.settings import SIGNAL_PARAMETERS

import numpy as np

save_folder = "./.saved_models/style_fa/"

blacklist = ['003_S_4288_S142486', '3169_BL_01',
             '3169_V04_00', '3168_BL_00', '3167_SC_01']

paths, sites_dict = get_paths_SIMON()

path_train, path_validation, path_test = train_test_val_split(
    paths,
    test_proportion=10 / 30,
    validation_proportion=10 / 30,
    # balanced_classes=[0, 1, 2],
    max_combined_size=30,
    blacklist=blacklist)

print("Train dataset size :", len(path_train))
print("Test dataset size :", len(path_test))
print("Validation dataset size", len(path_validation))

with open(save_folder + "train_val_test.txt", "w") as output:
    output.write(str([x['name'] for x in path_train]) + '\n')
    output.write(str([x['name'] for x in path_validation]) + '\n')
    output.write(str([x['name'] for x in path_test]))


SIGNAL_PARAMETERS['overlap_coeff'] = 1

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

np.save(save_folder + 'mean_std.npy',
        [train_dataset.mean, train_dataset.std])


net = ENet(sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=[16, 32, 64, 128],
           encoder_relu=False,
           decoder_relu=True)

sh_order = SIGNAL_PARAMETERS['sh_order']
ncoef = int((sh_order + 2) * (sh_order + 1) / 2)

feat_net = AdversarialNet(in_dim=ncoef,
                          out_dim=len(sites_dict),
                          num_filters=4,
                          nb_layers=5,
                          embed_size=128,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          gtab=validation_dataset.data[0]['gtab'],
                          sh_order=sh_order,
                          mean=train_dataset.mean,
                          std=train_dataset.std,
                          return_dict_layers=True)

net = net.to("cuda")
feat_net = feat_net.to("cuda")

B, _ = get_B_matrix(validation_dataset.data[0]['gtab'],
                    SIGNAL_PARAMETERS['sh_order'])

TRAINER_PARAMETERS = {
    "optimizer_parameters": {
        "autoencoder": {
            "lr": 0.001,
            "weight_decay": 1e-8,
        },
        "features": {
            "lr": 0.001,
            "weight_decay": 1e-8,
        }
    },
    "loss_specs": {
        "autoencoder": [
            {
                "type": "mse",
                "parameters": {},
                "coeff": 1,
            },
            {
                "type": "dwi_mse",
                "parameters": {"B": B,
                               "mean": train_dataset.mean,
                               "std": train_dataset.std,
                               "voxels_to_take": "all"},
                "coeff": 1,
            }],
        "style": [],
        "features": [
            {
                "type": "cross_entropy",
                "parameters": {},
                "coeff": 1,
            }]
    },
    "metrics": {
        "autoencoder": ["acc", "mse"],
        "features": ["accuracy"]
    },
    "metric_to_maximize": {
        "autoencoder": "mse",
        "features": "accuracy"
    },
    "patience": 100,
    "save_folder": save_folder,
}

style_trainer = StyleTrainer(
    net,
    feat_net,
    **TRAINER_PARAMETERS
)

# Train both networks each epoch

feat_net, adv_metrics = style_trainer.train_feat_net(train_dataset,
                                                     validation_dataset,
                                                     coeff_fake=0,
                                                     num_epochs=20,
                                                     batch_size=64,
                                                     validation=True)

style_trainer.feat_net = feat_net

validation_features = style_trainer.feat_net.predict_dataset(
    validation_dataset)


def flatten_dict(dic):
    return {k: np.concatenate([v[k] for v in dic.values()])
            for k in dic[list(dic.keys())[0]]}


validation_features = flatten_dict(validation_features)

# target_layers = [name for name in validation_features.keys() if "feat" in name]
target_layers = ['dense_feat_1',
                 'conv_feat_2', 'conv_feat_3']

target_features = {layer: np.mean(validation_features[layer], axis=0)[None]
                   for layer in target_layers}
layers_coeff = {name: 1. for name in target_layers}
print(target_layers)
style_trainer.set_style_loss([
    {"type": "style",
     "parameters": {"target_features": target_features,
                    "layers_coeff": layers_coeff},
     "coeff": 1e5,
     }])

net, metrics = style_trainer.train(train_dataset,
                                   validation_dataset,
                                   num_epochs=50,
                                   batch_size=16,
                                   validation=True)

style_trainer.net = net

test_dataset = SHDataset(path_test,
                         patch_size=SIGNAL_PARAMETERS["patch_size"],
                         signal_parameters=SIGNAL_PARAMETERS,
                         transformations=None,
                         normalize_data=True,
                         mean=train_dataset.mean,
                         std=train_dataset.std,
                         n_jobs=8,
                         cache_dir="./")


metrics_real = style_trainer.validate_feat(test_dataset, adversarial=False)
print('metrics_real:', metrics_real)

# Reset feat_net and train it on the fake patches

feat_net = AdversarialNet(in_dim=ncoef,
                          out_dim=len(sites_dict),
                          num_filters=4,
                          nb_layers=4,
                          embed_size=256,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          gtab=validation_dataset.data[0]['gtab'],
                          sh_order=sh_order,
                          mean=train_dataset.mean,
                          std=train_dataset.std,
                          return_dict_layers=True)

feat_net = feat_net.to("cuda")
style_trainer.feat_net = feat_net

feat_net, adv_metrics = style_trainer.train_feat_net(train_dataset,
                                                     validation_dataset,
                                                     coeff_fake=1,
                                                     num_epochs=20,
                                                     batch_size=16,
                                                     validation=True)

style_trainer.feat_net = feat_net

generated_dataset = AdversarialDataset(test_dataset, style_trainer.net)
generated_dataset.names = [x for x in generated_dataset.names if 'pred' in x]

metrics_fake = style_trainer.validate_feat(generated_dataset,
                                           adversarial=False)

print('metrics_fake:', metrics_fake)
