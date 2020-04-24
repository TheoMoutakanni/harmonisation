ADNI_PATH = '/home/theo/Documents/Data/'#'/media/theo/285EDDF95EDDC02C/Users/Public/Documents/ADNI'
#'/home/theo/Documents/Data/'
SIGNAL_PARAMETERS = {
    'patch_size': [16, 16, 16],
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


DATASET_PARAMETERS = {
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
        "autoencoder": {
            "type": "mse",
            "parameters": {}
        },
        "adversarial": {
            "type": "bce",
            "parameters": {}
        }
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
