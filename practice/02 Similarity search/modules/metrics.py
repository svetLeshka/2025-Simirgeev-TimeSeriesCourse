import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    for index in range(0, len(ts1)):
        ed_dist += (ts1[index] - ts2[index]) ** 2

    return np.sqrt(ed_dist)


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0
    scalar_multiply = np.dot(ts1, ts2)
    n = len(ts1)
    ut1 = np.sum(ts1) / n
    ut2 = np.sum(ts2) / n
    st1 = np.sqrt(np.sum(ts1 ** 2) / n - ut1 ** 2)
    st2 = np.sqrt(np.sum(ts2 ** 2) / n - ut2 ** 2)

    norm_ed_dist = np.sqrt(np.abs(2 * len(ts1) * (1 - (scalar_multiply - n * ut1 * ut2) / (n * st1 * st2))))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n, m = len(ts1), len(ts2)
    r = max(r * n, abs(n - m))
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m + 1, i + r + 1)):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m]
