from sgpykit.util import matlab
from sgpykit.util.create_sparse_grid_construct import create_sparse_grid_construct


def create_sparse_grid_multiidx_set(C, knots, lev2knots, S2=None, base=0):
    """
    Create a sparse grid starting from a multiindex-set rather than from a rule IDXSET(I) <= W.

    This function produces a sparse grid starting from a multiindex-set rather than
    from a rule IDXSET(I) <= W.

    Parameters
    ----------
    C : array_like
        The multiindex set C. C must be in lexicographic order and admissible.
    knots : callable or list of callable
        Function(s) to generate knots for each dimension.
    lev2knots : callable
        Function to map levels to number of knots.
    S2 : dict, optional
        Another sparse grid. If provided, the function tries to recycle tensor grids
        from S2 to build those of S instead of recomputing them. This can be helpful
        whenever sequences of sparse grids are generated. Note that *NO* check will be
        performed whether S2 was generated with the same lev2knots as the one given as
        input. S2 can also be empty, S2=[].
    base : int, optional
        if matrix C is using 1-based indexing

    Returns
    -------
    S : dict
        The sparse grid structure.
    C : array_like
        The multiindex set C, possibly modified.

    See Also
    --------
    check_set_admissibility : For admissibility check.
    create_sparse_grid : For further information on KNOTS, LEV2KNOTS, and on the sparse grid data structure S.
    """

    if not matlab.issorted(C, 'rows'):
        raise ValueError('The multiindex set C is not sorted.')

    N = C.shape[1]
    S, C = create_sparse_grid_construct(C, N, knots, lev2knots, S2, base)

    return S, C
