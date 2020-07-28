import numpy as np
import torch

from harmonisation.utils import rolling_window


def xyz_to_batch(signal, patch_size, overlap_coeff=1):
    """Convert the signal of shape (x, y, z, sh) to batches of size
    (batch, patch_x, patch_y, patch_z, sh) splitted into patches
    Return the batched signal and the number of patches by axis as tuple
    """

    coeff_size = signal.shape[-1]

    # Create the patches
    # (x, y, z, sh, patch_x, patch_y, patch_z)

    if torch.is_tensor(signal):
        signal_as_batch = signal.unfold(
            0, int(patch_size[0]), patch_size[0] // overlap_coeff).unfold(
            1, int(patch_size[1]), patch_size[1] // overlap_coeff).unfold(
            2, int(patch_size[2]), patch_size[2] // overlap_coeff)
    else:
        window = patch_size + [0]
        steps = [x // overlap_coeff for x in patch_size] + [1]
        signal_as_batch = rolling_window(signal, window=window, asteps=steps)

    number_of_patches = signal_as_batch.shape[:3]

    # Flatten the x,y,z axes to make a new batch axis and permute sh axis
    # (batch, patch_x, patch_y, patch_z, sh)
    signal_as_batch = signal_as_batch.reshape(
        -1, coeff_size, patch_size[0], patch_size[1], patch_size[2]
    )

    if torch.is_tensor(signal):
        signal_as_batch = signal_as_batch.permute((0, 2, 3, 4, 1))
    else:
        signal_as_batch = signal_as_batch.transpose((0, 2, 3, 4, 1))

    return signal_as_batch, number_of_patches


def batch_to_xyz(dmri_batch, real_size, empty=None, overlap_coeff=1,
                 remove_border=None):
    """Convert the signal of shape (batch, patch_x, patch_y, patch_z, sh)
    to a signal in contiguous coordinates (x, y, z, sh)
    Return the signal in contiguous coordinates

    Empty is an array of boolean to filter empty patches who where removed

    If overlap_coeff > 1, then each value will be the mean of all predicted
    values at the same xyz coordinate

    If regularization_sigma is not None, each value in a patch will be
    weighted by a 3D unnormalized gaussian kernel
    So if overlap_coef > 1, the voxels wich are on the borders of some patches
    will have less impact on the mean.
    """

    batch_size, patch_x, patch_y, patch_z, sh_coeff = dmri_batch.shape

    # Create indexes to know where each value is in xyz coordinates
    # idx is of the same shape (minus sh dim) than dmri_batch
    idx = np.arange(np.prod(real_size)).reshape(*real_size, 1)

    if torch.is_tensor(dmri_batch):
        idx = torch.LongTensor(idx)

    idx, _ = xyz_to_batch(idx,
                          [patch_x, patch_y, patch_z],
                          overlap_coeff)

    if empty is not None:
        idx = idx[~empty]

    number_of_voxels = np.prod(idx.shape[:-1])

    if torch.is_tensor(dmri_batch):
        dmri_flat = torch.zeros((np.prod(real_size), sh_coeff),
                                dtype=dmri_batch.dtype)

        # Get the weigths
        if remove_border is not None and remove_border > 0:
            weigths = torch.ones((patch_x, patch_y, patch_z))
            weigths[:remove_border] = 0
            weigths[-remove_border:] = 0
            weigths[:, :remove_border] = 0
            weigths[:, -remove_border:] = 0
            weigths[:, :, :remove_border] = 0
            weigths[:, :, -remove_border:] = 0
            weigths = weigths.repeat(batch_size, 1, 1)
            weigths = weigths.reshape(number_of_voxels, 1)
        else:
            weigths = torch.ones((number_of_voxels, 1))

        # Sum all values of dmri_batch at each xyz coordinates according to idx
        dmri_batch = dmri_batch.reshape(number_of_voxels, sh_coeff)

        dmri_flat.index_add_(0,
                             idx.reshape(number_of_voxels),
                             dmri_batch * weigths)

        # Get the number of predicted value per xyz coordinate
        NB = torch.zeros(np.prod(real_size), 1)
        NB.index_add_(0,
                      idx.reshape(number_of_voxels),
                      weigths)
        NB[NB == 0] = 1

    else:
        dmri_flat = np.zeros((np.prod(real_size), sh_coeff),
                             dtype=dmri_batch.dtype)

        # Get the weigths
        if remove_border is not None:
            weigths = np.ones((patch_x, patch_y, patch_z))
            weigths[:remove_border] = 0
            weigths[-remove_border:] = 0
            weigths[:, :remove_border] = 0
            weigths[:, -remove_border:] = 0
            weigths[:, :, :remove_border] = 0
            weigths[:, :, -remove_border:] = 0
            weigths = np.repeat(weigths[None, ...], batch_size, 0)
            weigths = weigths.reshape(number_of_voxels, 1)
        else:
            weigths = np.ones((number_of_voxels, 1))

        # Sum all values of dmri_batch at each xyz coordinates according to idx
        dmri_batch = dmri_batch.reshape(number_of_voxels, sh_coeff)

        np.add.at(dmri_flat,
                  idx.reshape(number_of_voxels),
                  dmri_batch * weigths)

        # Get the number of predicted value per xyz coordinate
        NB = np.zeros((np.prod(real_size), 1))
        np.add.at(NB,
                  idx.reshape(number_of_voxels),
                  weigths)
        NB[NB == 0] = 1

    # Compute mean and reshape
    dmri_xyz = (dmri_flat / NB).reshape(*real_size, sh_coeff)

    return dmri_xyz
