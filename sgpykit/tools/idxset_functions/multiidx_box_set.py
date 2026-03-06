import numpy as np

from sgpykit.util.checks import is_number_or_list_of_numbers_nonnegative
from sgpykit.util.misc import generate_multi_indices

def multiidx_box_set(shape, min_idx=0):
    """
    Generate a set of multi-dimensional box indices up to a given shape.

    Given an index `shape`, generates `C_with`, the box indices set up to that `shape`.
    `min_idx` is either 0 or 1. `C_without` is `C_with` without `shape` itself.

    Parameters
    ----------
    shape : array_like
        The shape of the box. Can be a scalar or a list/array of non-negative integers.
    min_idx : int
        Minimum index value, either 0 or 1.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - `C_with` : ndarray
            The full set of multi-indices up to `shape`.
        - `C_without` : ndarray
            The set of multi-indices without the last index.

    Notes
    -----
    If `shape` is a column vector with more than one entry, it is converted to a row vector
    with all entries equal to the minimum value of `shape`.
    """
    assert is_number_or_list_of_numbers_nonnegative(shape)
    assert min_idx == 0 or min_idx == 1

    shape = np.asarray(shape)
    assert shape.ndim < 3

    if shape.ndim == 2 and shape.shape[0] > 1:  # if it is a column vector
        # the column vector will converted to a row vector with all entries=min(shape)
        shape = np.ones(shape.size, dtype="int64") * np.min(shape)
    else:
        shape = shape.ravel()  # contiguous flattened array
    ## NOTE: rewritten implementation, does not use multiidx_gen as matlab code did
    result = generate_multi_indices(shape, min_idx)
    return result, result[:-1]
