from harmonisation.datasets import SHDataset
from harmonisation.utils import get_paths_ADNI, train_test_split
from harmonisation.trainers import BaseTrainer, AdversarialTrainer
from harmonisation.models import ENet, AdversarialNet


from harmonisation.settings import SIGNAL_PARAMETERS

blacklist = ['052_S_4959_S191689', '127_S_5200_S201964', '003_S_4555_S144329',
             '127_S_4844_S227064', '127_S_4148_S267784', '126_S_4896_S173777',
             '052_S_4945_S210328']
paths = get_paths_ADNI()
path_train, path_validation, path_test = train_test_split(
    paths,
    test_proportion=0,
    validation_proportion=0.2,
    max_combined_size=10)
# blacklist=blacklist)

print("Train dataset size :", len(path_train))
print("Test dataset size :", len(path_test))
print("Validation dataset size", len(path_validation))

# path_train = get_paths_ADNI(['003_S_0908_S210038'])
# path_test = get_paths_ADNI(['003_S_1074_S256382'])


train_dataset = SHDataset(path_train,
                          patch_size=SIGNAL_PARAMETERS["patch_size"],
                          signal_parameters=SIGNAL_PARAMETERS,
                          transformations=lambda x: x + 1e-8,
                          n_jobs=4,
                          cache_dir="./")

test_dataset = SHDataset(path_validation,
                         patch_size=SIGNAL_PARAMETERS["patch_size"],
                         signal_parameters=SIGNAL_PARAMETERS,
                         transformations=lambda x: x + 1e-8,
                         mean=train_dataset.mean,
                         std=train_dataset.std,
                         n_jobs=4,
                         cache_dir="./")


# net = AutoEncoderBase(patch_size=[12, 12, 12],
#                       mean_data=train_dataset.mean,
#                       std_data=train_dataset.std,
#                       sh_order=4,
#                       pdrop=0.2)

net = ENet(patch_size=SIGNAL_PARAMETERS["patch_size"],
           sh_order=4,
           embed=128,
           encoder_relu=False,
           decoder_relu=True)

adv_net = AdversarialNet(patch_size=SIGNAL_PARAMETERS["patch_size"],
                         sh_order=4,
                         embed=128,
                         encoder_relu=True)

net = net.to("cuda")
adv_net = adv_net.to("cuda")

# trainer = BaseTrainer(
#     net,
#     optimizer_parameters={
#         "lr": 0.01,
#         "weight_decay": 1e-8,
#     },
#     loss_specs={
#         "type": "mse",
#         "parameters": {}
#     },
#     metrics=["acc", "mse"],
#     metric_to_maximize="mse",
#     patience=100,
#     save_folder=None,
# )

adv_trainer = AdversarialTrainer(
    net,
    adv_net,
    optimizer_parameters={
        "autoencoder": {
            "lr": 0.001,
            "weight_decay": 1e-8,
        },
        "adversarial": {
            "lr": 0.001,
            "weight_decay": 1e-8,
        }
    },
    loss_specs={
        "autoencoder": {
            "type": "mse",
            "parameters": {}
        },
        "adversarial": {
            "type": "bce",
            "parameters": {}
        }
    },
    metrics={
        "autoencoder": ["acc", "mse"],
        "adversarial": ["accuracy"]
    },
    metric_to_maximize={
        "autoencoder": "acc",
        "adversarial": "accuracy"
    },
    patience=100,
    save_folder=None,
)


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
