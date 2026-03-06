import math
from functools import lru_cache

import numpy as np


def fast_TD_set(N, w, base=0):
    """
    Generate the total-degree multi-index set TD(w) in N dimensions.

    Parameters
    ----------
    N : int
        Number of dimensions.
    w : int
        Total degree.
    base : int, optional
        Base index to offset the multiindices (default is 0).

    Returns
    -------
    ndarray or int
        The multiindex set as a numpy array (or scalar if TDsize == 1 and N == 1).
    """
    # Initialize the size of the multiindex set
    TDsize = math.comb(N + w, N)

    # Generate all valid multi-indices recursively
    I = np.array(list(_generate_multiindices(N, w)))

    # Sort rows to get lexicographic order
    I = I[np.lexsort(I.T[::-1])]

    I += base

    # Handle single element case
    if TDsize == 1 and N == 1:
        return I[0, 0]
    return I

@lru_cache(maxsize=None)
def _generate_multiindices(N, w):
    """
    Recursively generate all multi-indices where the sum is ≤ w.
    Uses memoization to avoid redundant computations.
    """
    if N == 1:
        return [(i,) for i in range(w + 1)]
    else:
        indices = []
        for i in range(w + 1):
            for rest in _generate_multiindices(N - 1, w - i):
                indices.append((i,) + rest)
        return indices