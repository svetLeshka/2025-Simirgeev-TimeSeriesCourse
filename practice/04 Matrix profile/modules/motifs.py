import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    m = 165
    excl_zone = int(np.ceil(m / 2))
    zoned_matrix_profile = np.copy(matrix_profile['mp'])
    
    for i in range(top_k):
        min_index = np.argmin(zoned_matrix_profile)
        left_index = matrix_profile['mpi'][min_index] if matrix_profile['mpi'][min_index] < min_index else min_index
        right_index = matrix_profile['mpi'][min_index] if left_index == min_index else min_index
        motifs_idx.append(np.array([left_index, right_index]))
        motifs_dist.append(np.array([matrix_profile['mp'][left_index], matrix_profile['mp'][right_index]]))
        zoned_matrix_profile = apply_exclusion_zone(zoned_matrix_profile, min_index, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
