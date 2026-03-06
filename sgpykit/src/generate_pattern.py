import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_pattern(m):
    """
    Generate a pattern matrix for the Cartesian product of sequences.

    This function generates a matrix that can be used to create the Cartesian product
    of sequences {1, 2, ..., m1} x {1, 2, ..., m2} x ... x {1, 2, ..., mN}.

    Parameters
    ----------
    m : array_like
        Input array of integers representing the lengths of the sequences.

    Returns
    -------
    pattern : ndarray
        Matrix of indices for the Cartesian product. The shape is (N, prod(m)),
        where N is the length of m.

    Notes
    -----
    - The function expects MATLAB 1-based indexing and returns 0-based indexing.
    - The algorithm works one direction at a time, generating the n-th row of the pattern
      by repeating a base vector q times.

    Examples
    --------
    >>> generate_pattern([3, 2, 2])
    array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=uint8)
    """
    N = len(m)
    # it is convenient from a computational point of view to generate the pattern as unsigned int to create the pattern
    # TODO: check how these downcasted data types affect the performance/memory footprint
    if np.amax(m) <= 255:  # uint8 max
        pattern = np.zeros((N, np.prod(m, dtype=np.int64)), dtype='uint8')
    elif np.amax(m) <= 65535:  # uint16
        logger.warning('SparseGKit:uint16: more than 255 points are asked in one direction, using uint16 to handle this')
        pattern = np.zeros((N, np.prod(m, dtype=np.int64)), dtype='uint16')
    else:
        logger.warning('SparseGKit:double: more than 65535 points are asked in one direction, using double precision to handle this')
        pattern = np.zeros((N, np.prod(m, dtype=np.int64)))

    # the algorithm works one direction at a time. at the n-th iteration the n-th row of pattern is generated,
    # by repeating q times the vector BASE, which containes itselt a sequence,
    # obtained repeating p times each number from j=1 to j=m(n), e.g.
    # generate_pattern([3 2 2])

    # pattern =

    #       1      2      3      1      2      3      1      2      3      1      2      3
    #       1      1      1      2      2      2      1      1      1      2      2      2
    #       1      1      1      1      1      1      2      2      2      2      2      2

    for k in range(N):
        if k > 0:
            p = np.prod(np.append(1, m[:k]))
        else:
            p = 1
        if k < N - 1:
            q = np.prod(np.append(m[k + 1:], 1))
        else:
            q = 1
        # length of base vector
        lb = int(p * m[k])
        base = np.zeros(lb)
        # generate base vector
        bb = 0
        # print(k, p, q, lb, m[k])
        for j in range(m[k]):
            base[bb:bb + p] = j
            bb = bb + p
        # repeat base vector
        pp = 0
        for j in range(q):
            pattern[k, pp:pp + lb] = base
            pp = pp + lb

    return pattern
