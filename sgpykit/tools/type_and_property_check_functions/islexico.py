from sgpykit.util import matlab


def islexico(a, b):
    """
    Check lexicographic order of vectors.

    Checks if vector `a` is lexicographically less than or equal to vector `b`.
    Returns True if `a` <= `b` in lexicographic sense, False otherwise.

    Parameters
    ----------
    a : array_like
        First vector for comparison.
    b : array_like
        Second vector for comparison.

    Returns
    -------
    bool
        True if `a` is lexicographically less than or equal to `b`, False otherwise.

    Examples
    --------
    >>> islexico([1, 2, 3], [1, 2, 5])
    True
    >>> islexico([1, 2, 3], [1, 2, 3])
    True
    >>> islexico([1, 2, 3], [1, 2, 1])
    False
    >>> islexico([2, 2, 3], [1, 7, 1])
    False
    >>> islexico([1, 7, 3], [1, 5, 3])
    False
    """
    __, v = matlab.find(a - b, 1)

    # return true if a==b (i.e., v is empty) or if the first component for which a and b differ is smaller in a
    # than in b
    isl = len(v) == 0 or v[0] < 0
    return isl
