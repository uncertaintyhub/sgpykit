import logging
import numpy as np

from sgpykit.util import matlab

logger = logging.getLogger(__name__)


def detect_insufficient_tolerance(pts, tol):
    """
    Check if the given tolerance is sufficient to detect identical points.

    For a matrix of points (one per row, as in mysortrows or lookup_merge_and_diff),
    computes an approximation of the support of the set of point in each direction
    (as max - min coord in each dir) and compares it with tol. If they are "same-sized"
    then tol is too large and a warning is thrown.

    Parameters
    ----------
    pts : numpy.ndarray
        Matrix of points, with each row representing a point.
    tol : float
        Tolerance value to check.

    Returns
    -------
    bool
        True if the tolerance is insufficient, False otherwise.
    """
    # is_unsuf = detect_insufficient_tolerance(pts,tol)

    N = pts.shape[2 - 1]
    is_unsuf = False
    for n in range(N):
        # check one column at a time. exploit the fact that unique returns a sorted sequence
        uu, _ = matlab.unique(pts[:, n])
        # if the tolerance is not at least 2 order of magnitude smaller than the support on the n-th direction
        # throw a warning
        if len(uu)>1 and np.abs(np.log10(uu[-1] - uu[0]) - np.log10(tol)) < 2:
            logger.warning('SparseGKit:TolNotEnough: tol does not seem small enough to detect identical points')
            is_unsuf = True
            return is_unsuf

    return is_unsuf
