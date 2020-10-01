"""
Train an autoencoder with an adversarial loss
"""

from collections import OrderedDict
from harmonisation.datasets import SHDataset, AdversarialDataset
from harmonisation.utils import (get_paths_PPMI, get_paths_ADNI, get_paths_SIMON,
                                 train_test_val_split)
from harmonisation.models.metric_module import FAModule, DWIModule, EigenModule
from harmonisation.trainers import BaseTrainer
from harmonisation.models import ENet, AdversarialNet

from harmonisation.settings import SIGNAL_PARAMETERS
from harmonisation.functions.shm import get_deconv_matrix

import numpy as np

save_folder = "./.saved_models/style_test/"

blacklist = ['003_S_4288_S142486', '3169_BL_01',
             '3169_V04_00', '3168_BL_00', '3167_SC_01']

paths, sites_dict = get_paths_SIMON()

path_train, path_validation, path_test = train_test_val_split(
    paths,
    test_proportion=5 / 30,
    validation_proportion=10 / 30,
    # balanced_classes=[0, 1, 2],
    max_combined_size=30,
    blacklist=blacklist,
    seed=42)

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
                               b0_mean=train_dataset.b0_mean,
                               b0_std=train_dataset.b0_std,
                               n_jobs=8,
                               cache_dir="./")

np.save(save_folder + 'mean_std.npy',
        [train_dataset.mean, train_dataset.std,
         train_dataset.b0_mean, train_dataset.b0_std])


net = ENet(sh_order=SIGNAL_PARAMETERS['sh_order'],
           embed=[32, 32, 32, 32],
           encoder_relu=False,
           decoder_relu=True)

# net, _ = ENet.load("./.saved_models/style_fa/" + '49_net.tar.gz')

sh_order = SIGNAL_PARAMETERS['sh_order']
ncoef = int((sh_order + 2) * (sh_order + 1) / 2)

gtab = validation_dataset.data[0]['gtab']

fodf_sh_order = 8
response = [15e-3, 4e-4, 4e-4, 850]

R, _, B_dwi = get_deconv_matrix(gtab, response, fodf_sh_order)
R = R.diagonal() * B_dwi

dwi_module = DWIModule(gtab=gtab,
                       sh_order=sh_order,
                       mean=train_dataset.mean,
                       std=train_dataset.std,
                       b0_mean=train_dataset.b0_mean,
                       b0_std=train_dataset.b0_std)
dwi_module = dwi_module.to("cuda")

eigen_module = EigenModule(gtab)
eigen_module = eigen_module.to("cuda")

fa_module = FAModule()
fa_module = fa_module.to("cuda")

modules = {'fa': fa_module, 'dwi': dwi_module, 'evals': eigen_module}

sites_net = AdversarialNet(in_dim=20,
                           out_dim=len(sites_dict),
                           num_filters=4,
                           nb_layers=4,
                           embed_size=256,
                           patch_size=SIGNAL_PARAMETERS["patch_size"],
                           spectral_norm=True)

# sites_net, _ = AdversarialNet.load("./.saved_models/style_test/sites_55.tar.gz")

discriminator_net = AdversarialNet(in_dim=20,
                                   out_dim=1,
                                   num_filters=4,
                                   nb_layers=4,
                                   embed_size=256,
                                   patch_size=SIGNAL_PARAMETERS["patch_size"],
                                   spectral_norm=True)

# sites_net = AdversarialNet.load(save_folder + '49_sites_net.tar.gz')

net = net.to("cuda")
sites_net = sites_net.to("cuda")
discriminator_net = discriminator_net.to("cuda")

TRAINER_PARAMETERS = {
    "optimizer_parameters": {
        "autoencoder": {
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
        "sites": {
            "lr": 0.001,
            "weight_decay": 1e-3,
            "momentum": 0.9,
            "nesterov": True,
        },
        "discriminator": {
            "lr": 0.001,
            "weight_decay": 1e-5,
            "momentum": 0.9,
            "nesterov": True,
        },
    },
    "scheduler_parameters": {
        "autoencoder": {
            "base_lr": 1e-3,
            "max_lr": 1e-2,
            "step_size_up": 1000,
            "cycle_momentum": False,
        },
        "sites": {
            "base_lr": 1e-3,
            "max_lr": 1e-2,
            "step_size_up": 1000,
        },
        "discriminator": {
            "base_lr": 1e-3,
            "max_lr": 1e-2,
            "step_size_up": 1000,
        },
    },
    "modules": modules,
    "loss_specs": {
        "autoencoder": [
            {
                "type": "mse",
                "inputs": [
                    {"name": "sh", "net": "autoencoder"},
                    {"name": "sh"},
                    {"name": "mask"}],
                "parameters": {},
                "coeff": 1.,
            },
            {
                "type": "mse",
                "inputs": [
                    {"name": "dwi", "from": "autoencoder"},
                    {"name": "dwi"},
                    {"name": "mask"}],
                "parameters": {},
                "coeff": 1e-4,
            },
            {
                "type": "mse",
                "inputs": [
                    {"name": "mean_b0", "net": "autoencoder"},
                    {"name": "mean_b0"},
                    {"name": "mask"}],
                "parameters": {},
                "coeff": 10.,
            },
            {
                "type": "mse",
                "inputs": [
                    {"name": "fa", "from": "autoencoder"},
                    {"name": "fa"},
                    {"name": "mask"}],
                "parameters": {},
                "coeff": 10.,
            },
            # {
            #     "type": "mse_dwi",
            #     "inputs": ["fodf_sh_fake", "dwi_fake", "mask"],
            #     "parameters": {"B": R, "where_b0": gtab.b0s_mask},
            #     "coeff": 1e-6,
            # },
            # {
            #     "type": "negative_fodf",
            #     "inputs": ["fodf_sh_fake", "mask"],
            #     "parameters": {"gtab": gtab,
            #                    "response": response,
            #                    "sh_order": fodf_sh_order,
            #                    "lambda_": 1,
            #                    "tau": 0.1,
            #                    "size": 3,
            #                    "method": "center"},
            #     "coeff": 1e-6,
            # },
            # {
            #     "type": "l2_reg",
            #     "inputs": ["beta"],
            #     "parameters": {},
            #     "coeff": 1e-4,
            # },
            # {
            #     "type": "smooth_reg",
            #     "inputs": ["alpha"],
            #     "parameters": {},
            #     "coeff": 1.,
            # },
            # {
            #     "type": "smooth_reg",
            #     "inputs": ["beta"],
            #     "parameters": {},
            #     "coeff": 1.,
            # },
        ],
        "sites": [
            {
                "type": "cross_entropy",
                "inputs": [
                    {"name": "y_logits", "net": "sites"},
                    {"name": "site"}],
                "detach_input": True,
                "parameters": {},
                "coeff": 1,
            }],
        "discriminator": [
            {
                "type": "bce_logits_ones",
                "inputs": [
                    {"name": "y_logits", "net": "discriminator"}],
                "detach_input": True,
                "parameters": {},
                "coeff": 1,
            },
            {
                "type": "bce_logits_zeros",
                "inputs": [
                    {"name": "y_logits", "net": "discriminator",
                     "from": "autoencoder"}],
                "detach_input": True,
                "parameters": {},
                "coeff": 1,
            }]
    },
    "metrics_specs": {
        "autoencoder": [
            {
                "type": "acc",
                "inputs": [
                    {"name": "sh", "net": "autoencoder"},
                    {"name": "sh"},
                    {"name": "mask"}],
                "parameters": {}
            },
            {
                "type": "mse",
                "inputs": [
                    {"name": "sh", "net": "autoencoder"},
                    {"name": "sh"},
                    {"name": "mask"}],
                "parameters": {}
            }],
        "sites": [
            {
                "type": "accuracy",
                "inputs": [
                    {"name": "y_logits", "net": "sites"},
                    {"name": "site"}],
                "parameters": {}
            }],
        "discriminator": [
            {
                "type": "accuracy",
                "inputs": [
                    {"name": "y_logits", "net": "discriminator"}],
                "parameters": {"force_label": 1}
            },
            {
                "type": "accuracy",
                "inputs": [
                    {"name": "y_logits", "net": "discriminator",
                     "from": "autoencoder"}],
                "parameters": {"force_label": 0}
            }],
    },
    "metric_to_maximize": {
        "autoencoder": {
            "agg_fun": "mean",
            "inputs": {"mse_sh": {}}
        },
        "sites": {
            "agg_fun": "mean",
            "inputs": {"accuracy_y_logits": {"net": "sites"}}
        },
        "discriminator": {
            "agg_fun": "mean",
            "inputs": {
                "accuracy_y_logits": {"net": "discriminator"},
                "accuracy_y_logits_autoencoder": {"net": "discriminator",
                                                  "from": "autoencoder"}
            }
        },
    },
    "patience": 100,
    "save_folder": save_folder,
}

style_trainer = BaseTrainer(
    {'autoencoder': net,
     'sites': sites_net,
     'discriminator': discriminator_net},
    **TRAINER_PARAMETERS
)

# nets, adv_metrics = style_trainer.train(
#     ['sites'],
#     train_dataset,
#     validation_dataset,
#     num_epochs=60,
#     batch_size=192,
#     validation=True)
# sites_net = nets['sites']

# nets, adv_metrics = style_trainer.train(
#     ['discriminator'],
#     train_dataset,
#     validation_dataset,
#     num_epochs=5,
#     batch_size=16,
#     validation=True)
# discriminator_net = nets['discriminator']


# sites_net_pred = style_trainer.adversarial_net['sites'].predict_dataset(
#     validation_dataset)


# def flatten_dict(dic):
#     return {k: np.concatenate([v[k] for v in dic.values()])
#             for k in dic[list(dic.keys())[0]]}


# validation_features = flatten_dict(sites_net_pred)

# # target_layers = [name for name in validation_features.keys() if "feat" in name]
# target_layers = ['dense_feat_1',
#                  'conv_feat_2',
#                  'conv_feat_3']

# target_features = {layer: np.mean(validation_features[layer], axis=0)[None]
#                    for layer in target_layers}
# layers_coeff = {name: 1 / len(target_layers) for name in target_layers}
# print(target_layers)

style_losses = []

# style_losses += [
#     {"type": "gram",
#      "inputs": [layer + '_fake_sites'],
#      "parameters": {"target_features": target_features[layer],
#                     "layers_coeff": layers_coeff[layer]},
#      "coeff": 1e5,
#      }
#     for layer in target_layers]

# style_losses += [
#     {"type": "feature",
#      "inputs": [layer + '_fake_sites'],
#      "parameters": {"target_features": target_features[layer],
#                     "layers_coeff": layers_coeff[layer]},
#      "coeff": 10.,
#      }
#     for layer in target_layers]

style_losses += [
    {
        "type": "cross_entropy",
        "inputs": [
            {"name": "y_logits", "net": "sites",
             "from": "autoencoder", "recompute": True},
            {"name": "site"}],
        "parameters": {"smoothing": 0.1},
        "coeff": -100.,
    },
    {
        "type": "bce_logits_ones",
        "inputs": [
            {"name": "y_logits", "net": "discriminator",
             "from": "autoencoder", "recompute": True}
        ],
        "parameters": {},
        "coeff": 20.,
    }
]

style_trainer.set_loss({'autoencoder': style_losses}, add=True)
style_trainer.set_loss(
    {'sites':
     [
         # {
         #     "type": "cross_entropy",
         #     "inputs": OrderedDict({
         #            "y_logits": {"net": "sites"},
         #            "site": {},
         #        }),
         #     "parameters": {"smoothing": 0.1},
         #     "coeff": 1,
         # },
         {
             "type": "cross_entropy",
             "inputs": [
                 {"name": "y_logits", "net": "sites", "from": "autoencoder"},
                 {"name": "site"}],
             "detach_input": True,
             "parameters": {},
             "coeff": 1,
         }]
     },
    add=False,
)
style_trainer.set_metric(
    {"sites": [
        {
            "type": "accuracy",
                    "inputs": [
                        {"name": "y_logits", "net": "sites"},
                        {"name": "site"}],
            "parameters": {}
        },
        {
            "type": "accuracy",
                    "inputs": [
                        {"name": "y_logits", "net": "sites",
                         "from": "autoencoder"},
                        {"name": "site"}],
            "parameters": {}
        }],
     },
    add=False,
)

best_nets, metrics = style_trainer.train(
    ["sites", "discriminator", "autoencoder"],
    train_dataset,
    validation_dataset,
    num_epochs=60,
    batch_size=48,
    keep_best_net=False)


net = style_trainer.net
sites_net = style_trainer.adversarial_net['sites']
discriminator_net = style_trainer.adversarial_net['discriminator']

test_dataset = SHDataset(path_test,
                         patch_size=SIGNAL_PARAMETERS["patch_size"],
                         signal_parameters=SIGNAL_PARAMETERS,
                         transformations=None,
                         normalize_data=True,
                         mean=train_dataset.mean,
                         std=train_dataset.std,
                         b0_mean=train_dataset.b0_mean,
                         b0_std=train_dataset.b0_std,
                         n_jobs=8,
                         cache_dir="./")

style_trainer.set_adversarial_metric(
    {"sites": [
        {
            "type": "accuracy",
                    "inputs": ["y_logits_sites", "site"],
                    "parameters": {}
        },
        {
            "type": "accuracy",
                    "inputs": ["y_logits_fake_sites", "site"],
                    "parameters": {}
        }
    ]}
)

metrics_real = style_trainer.validate_adversarial(
    'sites',
    test_dataset,
    batch_size=16)

print('metrics_real:', metrics_real)

# Reset sites_net and train it on the fake patches

sites_net_test = AdversarialNet(in_dim=17,
                                out_dim=len(sites_dict),
                                num_filters=4,
                                nb_layers=4,
                                embed_size=256,
                                patch_size=SIGNAL_PARAMETERS["patch_size"],
                                modules=modules)

sites_net_test = sites_net_test.to("cuda")

style_trainer_test = StyleTrainer(
    net,
    {'sites': sites_net_test, 'discriminator': discriminator_net},
    **TRAINER_PARAMETERS
)

# Set loss to get y_logits from fake MRI
style_trainer_test.set_adversarial_loss(
    {'sites':
        [{
            "type": "cross_entropy",
            "inputs": ["y_logits_fake", "site"],
            "parameters": {},
            "coeff": 1,
        }]
     }
)

sites_net_test, _ = style_trainer_test.train_adversarial_net(
    ['sites'],
    train_dataset,
    validation_dataset,
    num_epochs=40,
    batch_size=8,
    validation=True)

sites_net_test = sites_net_test['sites']

metrics_fake = style_trainer_test.validate_adversarial(
    'sites',
    test_dataset,
    batch_size=16)

print('metrics_fake:', metrics_fake)
