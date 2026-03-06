import numpy as np


def islexico_tol(a, b, tol):
    """
    Check if vector a is lexicographically less-or-equal than vector b up to a numerical tolerance.

    Parameters
    ----------
    a : array_like
        First input vector.
    b : array_like
        Second input vector.
    tol : float
        Numerical tolerance for component-wise comparison.

    Returns
    -------
    bool
        True if a is lexicographically less-or-equal than b up to the given tolerance, False otherwise.

    Examples
    --------
    >>> ii = [1, 3]
    >>> jj = [1, 4]
    >>> islexico_tol(ii, jj, 1e-12)  # Returns True
    >>> ii = [1+1e-13, 3]
    >>> islexico_tol(ii, jj, 1e-12)  # Returns True
    >>> islexico_tol(ii, jj, 1e-14)  # Returns False
    """
    a = np.asarray(a)
    b = np.asarray(b)
    # Find the first index where a and b differ by more than tol
    diff = np.abs(a - b)
    first_diff_idx = np.argmax(diff > tol)

    # If no elements differ by more than tol, they are equal (return True)
    if diff[first_diff_idx] <= tol:
        return True

    # Otherwise, check if a[first_diff_idx] < b[first_diff_idx]
    return a[first_diff_idx] < b[first_diff_idx]
