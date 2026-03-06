import copy

import numpy as np


def tensor_to_sparse(T, idx=None):
    """
    Convert a tensor grid into a sparse grid structure.

    This function converts a tensor grid into a sparse grid structure of the same type
    as CREATE_SPARSE_GRID, by adding the missing fields.

    Parameters
    ----------
    T : Struct
        The tensor grid to be converted.
    idx : array_like, optional
        The index vector. If not provided, it will be computed based on the knots per dimension.

    Returns
    -------
    S : Struct or StructArray
        The sparse grid structure with added fields.

    Notes
    -----
    The function copies the fields of T into S and adds the following:
        - S.coeff = 1
        - S.idx = vector with idx(i) = length(T.knots_per_dim{i}) if idx is not provided
        - S.idx = idx if idx is provided
    """
    S = copy.copy(T)
    S.coeff = 1
    if idx:
        S.idx = idx
    else:
        N = len(S.knots_per_dim)
        idx = np.zeros(N, dtype=np.int64)
        for n in range(N):
            idx[n] = len(S.knots_per_dim[n])
        S.idx = idx

    return S
