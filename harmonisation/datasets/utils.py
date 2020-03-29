import numpy as np


def xyz_to_batch(signal, patch_size):
    """Convert the signal of shape (x, y, z, sh) to batches of size
    (batch, patch_x, patch_y, patch_z, sh) splitted into patches
    Return the batched signal and the number of patches by axis as tuple
    """

    coeff_size = signal.shape[-1]

    # Create the patches
    # (x, y, z, sh, patch_x, patch_y, patch_z)
    signal_as_batch = signal.unfold(
        0, int(patch_size[0]), int(patch_size[0])).unfold(
        1, int(patch_size[1]), int(patch_size[1])).unfold(
        2, int(patch_size[2]), int(patch_size[2]))

    number_of_patches = signal_as_batch.shape[:3]

    # Flatten the x,y,z axes to make a new batch axis and permute sh axis
    # (batch, patch_x, patch_y, patch_z, sh)
    signal_as_batch = signal_as_batch.reshape(
        -1, coeff_size, patch_size[0], patch_size[1], patch_size[2]
    ).permute((0, 2, 3, 4, 1)).contiguous()

    return signal_as_batch, number_of_patches


def batch_to_xyz(signal_as_batch, number_of_patches):
    """Convert the signal of shape (batch, patch_x, patch_y, patch_z, sh)
    to a signal in contiguous coordinates (x, y, z, sh)
    Return the signal in contiguous coordinates
    """
    number_of_patches = np.array(number_of_patches)

    # (batch, patch_size_x, patch_size_y, patch_size_z, nb_sh_coeff)
    batch_size, patch_x, patch_y, patch_z, sh_coeff = signal_as_batch.shape

    # reshape: (nb_x, nb_y, nb_z, patch_x, patch_y, patch_z, sh_coeff)
    dmri = signal_as_batch.reshape(
        *number_of_patches,
        patch_x, patch_y, patch_z,
        sh_coeff)
    # permute: (nb_x, patch_x, nb_y, patch_y, nb_z, patch_z, sh_coeff)
    dmri = dmri.permute((0, 3, 1, 4, 2, 5, 6))
    # reshape: (x, y, z, sh_coeff)
    dmri = dmri.reshape(*(number_of_patches * (patch_x, patch_y, patch_z)),
                        sh_coeff).contiguous()
    return dmri
