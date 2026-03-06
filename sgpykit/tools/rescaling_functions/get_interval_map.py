import numpy as np

from sgpykit.util.checks import is_nparray_of_numbers


def get_interval_map(a, b, type_):
    """
    Build a mapping function to shift sparse grids from a reference domain to a target domain.

    This function constructs and returns a callable that maps points from a reference domain
    (either the hypercube (-1,1)^N or standard Gaussian space) to a user-specified box or
    Gaussian product distribution.

    Parameters
    ----------
    a : numpy.ndarray
        Lower bounds of the target domain (one per dimension).
    b : numpy.ndarray
        Upper bounds of the target domain (one per dimension).
    type_ : str
        Type of mapping to construct. Supported values are:
        - 'uniform': maps points from (-1,1)^N to the box (a(1),b(1)) x ... x (a(N),b(N))
        - 'gaussian': maps points from standard Gaussian space to a product of Gaussians with
          means in `a` and standard deviations in `b`

    Returns
    -------
    callable
        A function that takes a matrix of points (one per column) in the reference domain and
        returns a matrix of points in the target domain. If `type_` is not supported, returns None.

    Notes
    -----
    The output function must be used as X = interval_map(T), where T is a matrix of points
    in the reference domain (one point per column), and X is a matrix of points in the target
    domain (one point per column).
    """
    assert is_nparray_of_numbers(a) and is_nparray_of_numbers(b) and isinstance(type_, str)
    a = np.ravel(a)  # get a 1d row vector view
    b = np.ravel(b)
    if 'uniform' == type_:
        # first we build D, resize matrix: it modifies a column vector "c" with components in 0,1
        # into a column vector "d" with components in ( 0 |b(1)-a(1)| ) x ( 0 |b(2)-a(2)| ) x ...
        D = np.diag(b - a)
        # next a translation vector: shifts a column vector "d" with components in ( 0 |b(1)-a(1)| ) x ( 0 |b(2)-a(2)| ) x ...
        # to "e" with components  in ( a(1) b(1) ) x ( a(2) b(2) ) x ...

        # finally, define the function
        fmap = lambda T: D @ (T + 1) / 2 + np.atleast_2d(a).T @ np.ones((1, T.shape[1]))
    else:
        if 'gaussian' == type_:
            # D is the stdev matrix
            D = np.diag(b)
            # shift vector, according to means

            # finally, define the function
            fmap = lambda T: D @ T + np.atleast_2d(a).T @ np.ones((1, T.shape[1]))
        else:
            # NOTE: added this else branch in sgpykit
            fmap = None

    return fmap
