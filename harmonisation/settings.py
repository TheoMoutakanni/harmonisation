PPMI_PATH = '/home/theo/Documents/Data/PPMI/'
ADNI_PATH = '/home/theo/Documents/Data/ADNI/'
SIMON_PATH = '/media/theo/285EDDF95EDDC02C/Users/Public/Documents/SIMON/CCNA/'

SIGNAL_PARAMETERS = {
    'patch_size': [32, 32, 32],
    'sh_order': 4,
    'overlap_coeff': 2,
    'processing_params': {
        'median_otsu_params': {
            'median_radius': 3,
            'numpass': 1,
        },
        'sh_params': {
            'basis_type': 'descoteaux07',
            'smooth': 0.006
        }
    }
}


TRAINER_PARAMETERS = {
    "optimizer_parameters": {
        "autoencoder": {
            "lr": 0.001,
            "weight_decay": 1e-8,
        },
        "adversarial": {
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
            }],
        "adversarial": [
            {
                "type": "cross_entropy",
                "parameters": {},
                "coeff": 1,
            }]
    },
    "metrics": {
        "autoencoder": ["acc", "mse"],
        "adversarial": ["accuracy"]
    },
    "metric_to_maximize": {
        "autoencoder": "acc",
        "adversarial": "accuracy"
    },
    "patience": 100,
    "save_folder": None,
}

# NETWORK_PARAMETERS = {
#     "patch_size": SIGNAL_PARAMETERS["patch_size"],
#     "sh_order": SIGNAL_PARAMETERS['sh_order'],
#     "embed": 128,
#     "encoder_relu": False,
#     "decoder_relu": True
# }
