import numpy as np

from sgpykit.util.misc import append


def multiidx_gen(N, rule, w, base=0, multiidx=None, MULTI_IDX=None):
    """
    Generate all multi-indexes of length N with elements such that rule(M_I) <= w.

    This function recursively explores the tree of all possible multi-indexes and
    stores the valid ones as rows of the matrix MULTI_IDX. The indices start from
    the specified base (either 0 or 1).

    Parameters
    ----------
    N : int
        Length of the multi-index.
    rule : callable
        Function that evaluates the rule for a given multi-index.
    w : float
        Threshold for the rule.
    base : int, optional
        Starting index for the multi-index elements (default is 0).
    multiidx : list, optional
        Current multi-index being constructed (default is None).
    MULTI_IDX : list or numpy.ndarray, optional
        Accumulator for valid multi-indexes (default is None).

    Returns
    -------
    numpy.ndarray or int
        Matrix of valid multi-indexes (or scalar if single element).

    Notes
    -----
    The function works recursively, exploring in depth the tree of all possible
    multi-indexes. The starting point is the empty multi-index: [], and MULTI_IDX
    is empty at the first call of the function. That's why the call from keyboard
    comes with [], [] as input argument: multiidx_gen(L,rule,w,[],[]).
    """
    # MULTI_IDX = multiidx_gen(N,rule,w,base,[],[])
    if multiidx is None:
        multiidx = []
    if MULTI_IDX is None:
        MULTI_IDX = []
    # calculates all multi indexes M_I of length N with elements such that rule(M_I) <= w.
    # M_I's are stored as rows of the matrix MULTI_IDX
    # indices will start from base (either 0 or 1)
    # multiidx_gen works recursively, exploring in depth the tree of all possible multiindexes.
    # the current multi-index is passed as 4-th input argument, and eventually stored in MULTI_IDX.
    # The starting point is the empty multiidx: [], and MULTI_IDX is empty at the first call of the function.
    # That's why the call from keyboard comes with [], [] as input argument: multiidx_gen(L,rule,w,[],[])

    if len(multiidx) != N:
        # recursive step: generates all possible leaves from the current node (i.e. all multiindexes with length le+1 starting from
        # the current multi_idx, which is of length le that are feasible w.r.t. rule)
        i = base
        while rule(append(multiidx, i)) <= w:
            # if [multiidx, i] is feasible further explore the branch of the tree that comes out from it.
            MULTI_IDX = multiidx_gen(N, rule, w, base, multiidx + [i], MULTI_IDX)
            i = i + 1

    #       for i=0:w
    #             if rule([multiidx, i]) <= w
    #                   # if [multiidx, i] is feasible further explore the branch of the tree that comes out from it.
    #                   MULTI_IDX = multiidx_gen(L,rule,w,[multiidx, i],MULTI_IDX);
    #             end
    #       end
    else:
        # base step: if the length of the current multi-index is L then I store it in MULTI_IDX  (the check for feasibility was performed in the previous call
        if len(MULTI_IDX) == 0:
            MULTI_IDX = [multiidx]
        else:
            MULTI_IDX = np.vstack((MULTI_IDX, multiidx))

    # handle single element case
    if len(multiidx) == 0:
        if np.atleast_1d(MULTI_IDX).size == 1:
            return np.array(MULTI_IDX).ravel()[0]
        else:
            return np.array(MULTI_IDX)
    return MULTI_IDX
