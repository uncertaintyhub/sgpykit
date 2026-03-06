import logging
import numpy as np

from sgpykit.src.detect_insufficient_tolerance import detect_insufficient_tolerance
from sgpykit.src.mysortrows import mysortrows

logger = logging.getLogger(__name__)


def lookup_merge_and_diff(pts_list, pts_list_old, tol=1e-14):
    """
    Identify points to compute, recycle, and discard between two sparse grids.

    This function compares two sets of points (from new and old sparse grids)
    and categorizes them based on their presence in both grids, using a specified
    tolerance to determine equality. It is used in adaptive sparse grid algorithms
    to efficiently manage point evaluations.

    Parameters
    ----------
    pts_list : numpy.ndarray
        Array of points from the new sparse grid (shape: N x d, where N is the number
        of points and d is the dimensionality).
    pts_list_old : numpy.ndarray
        Array of points from the old sparse grid (shape: N_old x d).
    tol : float, optional
        Tolerance for considering two points equal (default is 1e-14).

    Returns
    -------
    tocomp_list : numpy.ndarray
        Indices of points in pts_list that need to be evaluated (not present in old grid).
    recycle_list : numpy.ndarray
        Indices of points in pts_list that can be recycled from old grid.
    recycle_list_old : numpy.ndarray
        Corresponding indices in pts_list_old for recycled points.
    discard_list : numpy.ndarray
        Indices of points in pts_list_old that are no longer in the new grid.

    Notes
    -----
    - The function uses lexicographic sorting with tolerance to identify matching points.
    - For non-nested grids, some points from the old grid may be discarded.
    - The function performs several sanity checks to ensure correctness.
    """
    # [tocomp_list,recycle_list,recycle_list_old,discard_list] = lookup_merge_and_diff(pts_list,pts_list_old,tol)
    # looks for points of pts_list in pts_list_old using the same algorithm as reduce_sparse_grid. tol is
    # the tolerance for two points to be considered equal. discard_list is the list of points that no longer belong to the grid
    # (may happen for non-nested grids)

    N = pts_list.shape[0]
    N_old = pts_list_old.shape[0]
    # the list of indices of points to be evaluated. Init to its max length, will be cut after the search is over
    tocomp_list = np.zeros(N, dtype=np.int64)
    # if grid are not nested, not all of the old grid will be recycled. Thus,
    # we also need 2 list of indices of points to be recycled, storing the
    # positions  in the old grid and in the new one
    recycle_list_old = np.zeros(N_old, dtype=np.int64)
    recycle_list = np.zeros(N, dtype=np.int64)
    discard_list = np.zeros(N_old, dtype=np.int64)
    # first, merge the two lists
    Merged = np.vstack((pts_list_old, pts_list))
    # and a safety check: are we using a sufficiently fine tol when detecting identical points?
    detect_insufficient_tolerance(Merged, tol)
    # next, let's order the rows of Merged in lexicographic order, obtaining Sorted. If I use mysortrows then two rows like

    # [a b c d]
    # [a-t b c+t d]

    # are considered equal (if t < tol ) and therefore placed one after the other

    # sorter is an index vector that maps Merged into Sorted, i.e. Merged[sorter,:]==Sorted

    Sorted, sorter = mysortrows(Merged, tol)  ## same fix as reduce_sparse_grid mysortrows usage
    # I also need to remember which points come from pts_list and which from
    # pts_list_old, and from which original position.
    # Thus I create a flag vector [-1 -2 ... -N_old 1 2 .. N], i.e. positive
    # flags identify the new grid and negative ones the old grid, then I sort
    # it according to sorter

    flags = np.concatenate((-np.arange(1, N_old+1), np.arange(1, N+1)))
    flags_sorted = flags[sorter]

    # next I take the difference of two consecutive rows. if the difference is small, then the rows are the same, i.e. the knot is the same
    dSorted = np.diff(Sorted, axis=0)

    # I measure the difference with infty norm instead of L2 norm:
    # i take  the maximum component of each row (2 means "operate on columns"):
    #       max(abs(dSorted),[],2)
    # then I want to see which ones have this max bigger than tol
    #       diff_eq=max(abs(dSorted),[],2)>tol
    # this command returns a vector of True and false
    #       diff_eq=[1 1 0 1 1]
    # this means that the 2nd point is different from the 1st, the 3rd from the 2nd, but
    # the 4th is equal to the 3rd ( diff(3)=v(4)-v(3) ), hence in common between the
    # grids.
    diff_eq = np.any(np.abs(dSorted) > tol, axis=1)
    #diff_eq = np.amax(np.abs(dSorted), [], 2) > Tol
    # now I scroll diff_eq and sort out everything according to these rules:

    # --> if diff_eq(k)==0,

    # in this case either Sorted(k+1) is in the new grid and Sorted(k)
    # is in the old grid and both are equal or viceversa.

    # Therefore, the point in the new grid goes into recycle_list and the
    # old one in recycle_list_old.

    # Then I can skip the following (since it's equal and I have sorted it
    # already)

    # --> else, diff_eq(k)==1.

    # In this case, 4 cases are possible but actually only two matters

    # -----> both Sorted(k) and Sorted(k+1) comes from the old_grid.
    #   then, Sorted(k) is to discard

    # -----> both Sorted(k) and Sorted(k+1) comes from the new_grid.
    #   then, Sorted(k) is to compute

    # -----> Sorted(k) is new, Sorted(k+1) is old.
    #   then, Sorted(k) is to compute

    # -----> Sorted(k) is old, Sorted(k+1) is new.
    #   then, Sorted(k) is to discard

    i = 0  # scrolls recycle lists
    j = 0  # scrolls compute_list
    k = 0  # scrolls diff_eq

    L = len(diff_eq)
    discard = 0
    while k < L:

        if diff_eq[k]:  # Two consecutive entries are different
            if flags_sorted[k] > 0:
                tocomp_list[j] = flags_sorted[k]-1 # 0-based
                j += 1
            else:
                discard_list[discard] = -flags_sorted[k]-1 # 0-based
                discard += 1
            # then move to the following
            k += 1
        else:  # Two consecutive entries are equal
            if flags_sorted[k] > 0:
                recycle_list[i] = flags_sorted[k]-1           # 0-based
                recycle_list_old[i] = -flags_sorted[k + 1]-1  # 0-based
            else:
                recycle_list_old[i] = -flags_sorted[k]-1
                recycle_list[i] = flags_sorted[k + 1]-1
            i += 1
            # then I can skip the k+1 because I have already sorted it
            k += 2

    # need to handle the case k==L, since diff is 1 element shorter than sorted. Note
    # that
    # --> the node in Sorted[L,:] has been already taken care of inside the while loop
    # --> we only need to do something if the diff_eq(L)==1. Indeed, if
    # diff_eq(L)==0, then Sorted(L+1,:) is already taken care of (and in this
    # case the final value of k is L+2)
    if diff_eq[L - 1]:  # 0-based indexing
        if flags_sorted[L] > 0:
            tocomp_list[j] = flags_sorted[L]-1  # 0-based
            j += 1
        else:
            discard_list[discard] = -flags_sorted[L-1]-1 # 0-based
            discard += 1

    # show some statistics
    logger.debug(f'new evaluation needed: {j} recycled evaluations: {i}, discarded evaluations: {discard}')

    # remove the extra entries of tocomp_list, recycle_lists, discard_list. Pay attention to special cases
    if j != N:  # in this case there are no points to recycle and we have completely filled  tocomp_list
        if tocomp_list[j] != 0:
            raise RuntimeError('Failed sanity check: tocomp_list[j]~=0')
        tocomp_list = tocomp_list[:j]

    if i == N:
        logger.warning('the two grids are the same!')
    else:
        if i > N:
            raise RuntimeError('Failed sanity check: The code has detected more points to recycle than points in the new sparse grid! '
                            'Double check the values of tolerances used to detect identical points (both here and in reduce_sparse_grid) '
                            'and rerun the code. i>N')

        if recycle_list[i] != 0:
            raise RuntimeError('Failed sanity check: recycle_list[i]~=0')
        recycle_list = recycle_list[:i]

    if i != len(recycle_list_old):
        if recycle_list_old[i] != 0:
            raise RuntimeError('Failed sanity check: recycle_list_old[i]~=0')
        recycle_list_old = recycle_list_old[:i]

    if discard != N_old:
        if discard_list[discard] != 0:
            raise RuntimeError('Failed sanity check: discard_list[discard]~=0')
        discard_list = discard_list[:discard]

    # safety checks
    if i + discard != N_old:
        raise RuntimeError('Failed sanity check: The code has lost track of some points of the old grid, i+discard~=N_old')

    if len(recycle_list) != len(recycle_list_old):
        raise RuntimeError('Failed sanity check: mismatch between the two sets of recycling points. length(recycle_list)~=length(recycle_list_old)')

    if not np.array_equal(np.sort(np.concatenate((tocomp_list, recycle_list))), np.arange(0, N)):
        raise RuntimeError('Failed sanity check: The code has lost track of some points of the new grid, or some points from the old grid have been mistaken as points of the new grid. Double check the values of tolerances use to detect identical points (both here and in reduce_sparse_grid) and try to rerun the code.')

    return tocomp_list, recycle_list, recycle_list_old, discard_list
