ADNI_PATH = '/media/theo/285EDDF95EDDC02C/Users/Public/Documents/ADNI'

SIGNAL_PARAMETERS = {
    'patch_size': [32, 32, 12],
    'processing_params': {
        'median_otsu_params': {
            'median_radius': 3,
            'numpass': 1,
        },
        'sh_params': {
            'sh_order': 4,
            'smooth': 0.006,
        }
    }
}
