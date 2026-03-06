import numpy as np

from sgpykit.util import matlab
from sgpykit.util.checks import is_nparray_of_numbers
from sgpykit.util.misc import unique_rows_stable


def check_index_admissibility(idx, idx_set, sort_option=None):
    """
    Check if a multiindex is admissible with respect to an index set.

    Given a multiindex `idx` as a row vector, checks if it is admissible with
    respect to the index set `idx_set` (matrix with indices as rows). If it is
    admissible, returns `is_adm = True`. Otherwise, returns `is_adm = False`
    and the function will return the `completed_set` and a list of the indices
    added stored in `missing_set`.

    Parameters
    ----------
    idx : array_like
        Multiindex to check for admissibility.
    idx_set : array_like
        Matrix of indices (rows) representing the index set.
    sort_option : str, optional
        If 'sorting', returns `completed_set` and `missing_set` sorted in
        lexicographic order.

    Returns
    -------
    is_adm : bool
        True if the multiindex is admissible, False otherwise.
    completed_set : ndarray
        The completed index set after adding missing indices.
    missing_set : ndarray
        The list of indices added to make the set admissible.

    Notes
    -----
    The starting `idx_set` is assumed to be admissible, so its multi-indices
    will not be checked for admissibility. However, being admissible is not
    necessary for this function to work, since the core is `setdiff`, which does
    not assume any ordering.
    Assumes 0-based indexing.
    """
    is_adm = True
    completed_set = idx_set.copy()
    missing_set = np.empty(0, dtype=np.int64)
    # now check if idx is admissible. Note that if this is not the case, the indices needed
    # may not be included in idx_set, too !
    # So we add everything we need to check in a queue and keep on checking while the queue is empty

    # here's the queue
    the_queue = np.atleast_2d(idx)
    while not len(the_queue) == 0:

        # consider the first element of the stack
        i = the_queue[0,:]
        # build its needed set
        S = needed_set(i)
        # take the setdiff with idx_set. needed_rows is s.t. missing=needed_set[needed_rows,:]
        missing = matlab.setdiff(S, idx_set, 'rows')
        # if missing is not empty,
        if not len(missing) == 0:
            # the initial set was not admissible
            is_adm = False
            # missing had to be added to completed_set, missing_set, and the_queue.
            # I have to make sure I don't add duplicates,
            # and that the order of completed_set, missing_set, the_queue stays the same!
            # So I cannot use unique, since it automatically reorders rows
            # (and old versions of matlab like 2011b do not provide the 'stable'
            # option that deactivates the ordering).
            # setdiff(A,B,'rows') returns the rows from A that are not in B.
            # add missing to completed_set

            # completed_set = safe_vstack(completed_set, setdiff(missing, completed_set, 'rows'))
            # # add missing to missing set
            # if len(missing_set) == 0:
            #     missing_set = missing.copy()
            # else:
            #     missing_set = safe_vstack(missing_set, setdiff(missing, missing_set, 'rows'))
            # # add missing to the queue
            # adiff = setdiff(missing, the_queue, 'rows')
            # the_queue = safe_vstack(the_queue, adiff)
            # version with unique, does not work on R2011b
            # ----------------------------------
            # add missing to completed_set
            completed_set = unique_rows_stable(completed_set, missing)
            # add missing to missing set
            missing_set = unique_rows_stable(missing_set, missing)
            # add missing to the queue
            the_queue = unique_rows_stable(the_queue, missing)
        # delete current i from the queue
        the_queue = the_queue[1:, :]  ## without first row

    # if requested, sort in ascending lexicographic order
    if sort_option is not None:
        if sort_option == 'sorting':
            missing_set,_ = matlab.sortrows(missing_set)
            completed_set,_ = matlab.sortrows(completed_set)
        else:
            raise ValueError('unknown sorting option')

    return is_adm, completed_set, missing_set


def needed_set(idx):
    """
    Compute the indices of the form idx - e_j where e_j is the j-th N-dimensional unit vector.

    Parameters
    ----------
    idx : array_like
        Multiindex for which to compute the needed set.

    Returns
    -------
    S : ndarray
        Matrix of indices (rows) representing the needed set.

    Notes
    -----
    If `idx` has 1 inside, like [2 1 1], [3 1 2], etc., care has to be taken:
    the minimum value for indices is 1, so indices like [2 0 1], [2 1 0], etc. are
    not included in the set. This is handled by deleting all rows that contain 0.
    Note that 0 can be only in the main diagonal of needed_set.
    Assumes 0-based indexing.
    """
    assert is_nparray_of_numbers(idx)
    # S = needed_set(idx)

    # computes the indices of the form idx-e_j where e_j is the j-th N-dimensional unit vector,
    # and store them as rows of S

    N = len(idx)
    # I can build the set quickly with matrices operation: [idx; idx; ... ; idx] - eye(N)

    S = np.ones((N, 1)) * idx - np.eye(N)
    # if idx has 1 inside, like [2 1 1], [3 1 2] etc, care has to be taken: the minimum value for indices is
    # 1, so I don't have to check for [2 0 1], [2 1 0] etc to be in the set. I handle this deleting all rows
    # that contain 0. Note that 0 can be only in the main diagonal of needed_set

    D = np.diag(S)
    S = S[D >= 0, :] # ==0 due to 0-based indexing
    return S
