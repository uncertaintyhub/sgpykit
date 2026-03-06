import numpy as np

from sgpykit.tools.type_and_property_check_functions.islexico import islexico
from sgpykit.util import matlab


def find_lexicographic(lookfor, I, nocheck=None):
    """
    Find specific rows of a matrix that is sorted lexicographically.

    Parameters
    ----------
    lookfor : array_like
        The row vector to look for among the rows of I.
    I : array_like
        The matrix to search in, assumed to be sorted lexicographically.
    nocheck : str, optional
        If 'nocheck', the function does not check whether I is lexicographically sorted.
        This can be useful for speed purposes. Default is None.

    Returns
    -------
    found : bool
        True if the row vector is found, False otherwise.
    pos : int or None
        The row number of the found row vector, i.e., lookfor == I[pos, :]. If not found, pos is None.
    iter : int
        The number of iterations performed during the binary search.

    Notes
    -----
    The function performs a preliminary check whether I is actually lexicographically sorted.
    If nocheck is 'nocheck', this check is skipped.
    The function uses a binary search algorithm, which guarantees a cost of ~log(nb_idx).
    """
    if nocheck is not None and nocheck != 'nocheck':
        raise ValueError('unknown 3rd input')

    if nocheck != 'nocheck' and not matlab.issorted(I, 'rows'):
        raise ValueError('I is not lexicographically sorted')

    nb_idx = I.shape[0]
    if nb_idx == 0:
        return False, None, 0
    # we exploit the fact that I is sorted lexicographically and we proceed by binary search
    # which guarantees cost ~ log(nb_idx)

    # Basically, we start from the middle row, compare it with the index to be found, and
    # if our index is larger we make a step in the increasing direction (i.e. we look in the upper half
    # of the sorting), and viceversa. Of course, the step halves at each iteration:
    # therefore we necessarily terminate in ceil(log2(nb_idx)) steps at most

    # the position to compare against -- if found, this is the position to be returned

    ## REIMPLEMENTATION (0-based position)
    nb_idx = I.shape[0]
    if nb_idx == 0:
        return False, None, 0

    # Binary search initialization
    left, right = 0, nb_idx - 1
    iter_ = 0
    itermax = np.ceil(np.log2(nb_idx))

    while left <= right and iter_ <= itermax:
        iter_ += 1
        mid = (left + right) // 2
        jj = I[mid, :]

        if np.all(jj == lookfor):
            return True, mid, iter_  # mid is 0-based index for position
        elif islexico(jj, lookfor):
            left = mid + 1
        else:
            right = mid - 1

    return False, None, iter_
