import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    m = 96
    excl_zone = int(np.ceil(m / 2))
    zoned_matrix_profile = np.copy(matrix_profile['mp'])

    for i in range(top_k):
        max_index = np.argmax(zoned_matrix_profile)
        discords_idx.append(max_index)
        discords_dist.append(matrix_profile['mp'][max_index])
        discords_nn_idx.append(matrix_profile['mpi'][max_index])
        zoned_matrix_profile = apply_exclusion_zone(zoned_matrix_profile, max_index, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
