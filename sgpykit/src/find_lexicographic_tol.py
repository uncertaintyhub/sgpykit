import numpy as np

from sgpykit.tools.type_and_property_check_functions.islexico_tol import islexico_tol


def find_lexicographic_tol(lookfor, I, tol=1e-14):
    """
    Find specific rows of a matrix that is sorted lexicographically up to TOL.

    Finds a specific row vector in a matrix that is sorted lexicographically up to a
    given tolerance. The function uses binary search to efficiently locate the row.

    Parameters
    ----------
    lookfor : array_like
        The row vector to search for in the matrix.
    I : ndarray
        The matrix to search in, assumed to be sorted lexicographically.
    tol : float, optional
        The tolerance for testing equality between `lookfor` and rows of `I`.
        Default is 1e-14.

    Returns
    -------
    found : bool
        True if `lookfor` is found in `I` within the given tolerance, False otherwise.
    pos : int or list
        The row number of `lookfor` in `I` if found, otherwise an empty list.
    iter : int
        The number of iterations performed during the binary search.

    Notes
    -----
    The function does not perform a preliminary check whether `I` is actually
    lexicographically sorted. The binary search guarantees a cost of ~log(nb_idx).

    Examples
    --------
    >>> I,_ = sg.multiidx_box_set([4, 2], 1)
    >>> noise = 1e-13 * np.random.randn(*I.shape)
    >>> Inoisy = I + noise
    >>> print(sg.find_lexicographic_tol([4, 2], Inoisy, 1e-12))  # Should return True
    >>> print(sg.find_lexicographic_tol([4, 2], Inoisy, 1e-14))  # Should return False
    """
    # fixing optional inputs
    # we exploit the fact that I is sorted lexicographically and we proceed by binary search
    # which guarantees cost ~ log(nb_idx)
    # Basically, we start from the middle row, compare it with the index to be found, and
    # if our index is larger we make a step in the increasing direction (i.e. we look in the upper half
    # of the sorting), and viceversa. Of course, the step halves at each iteration:
    # therefore we necessarily terminate in ceil(log2(nb_idx)) steps at most
    # the position to compare against -- if found, this is the position to be returned

    ## REIMPLEMENTATION.
    nb_idx = I.shape[0]
    if nb_idx == 0:
        return False, [], 0

    # Binary search initialization
    left, right = 0, nb_idx - 1
    iter_ = 0
    itermax = np.ceil(np.log2(nb_idx))

    while left <= right and iter_ <= itermax:
        iter_ += 1
        mid = (left + right) // 2
        jj = I[mid, :]

        if np.all(np.isclose(jj, lookfor, atol=tol, rtol=0)):  # do not forget that rtol=0
            return True, mid, iter_  # mid is 0-based index for position
        elif islexico_tol(jj, lookfor, tol):
            left = mid + 1
        else:
            right = mid - 1

    return False, [], iter_
