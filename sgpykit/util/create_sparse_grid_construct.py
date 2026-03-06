import logging
import numpy as np

from sgpykit.src.find_lexicographic import find_lexicographic
from sgpykit.src.tensor_grid import tensor_grid
from sgpykit.util import matlab
from sgpykit.util.misc import apply_lev2knots, matlab_to_python_index

logger_ = logging.getLogger(__name__)


def create_sparse_grid_construct(C, N, knots, lev2knots, S2=None, base=0):
    """
    Construct a sparse grid from a given set of multi-indices.

    This function builds a sparse grid by combining tensor grids according to the
    combination technique. It optionally reuses tensor grids from a previously
    constructed sparse grid to improve efficiency.

    Parameters
    ----------
    C : ndarray
        Array of multi-indices (in 0 or 1-based index scheme) defining the sparse grid.
    N : int
        Number of dimensions.
    knots : callable or list of callable
        Function(s) to generate knots for each dimension.
    lev2knots : callable
        Function to convert level indices to number of knots.
    S2 : struct, optional
        Previously constructed sparse grid for tensor grid recycling.
    base : int, optional
        if matrix C is using 1-based indexing

    Returns
    -------
    S : struct
        Constructed sparse grid with fields:
        - knots : Cell array of knot coordinates for each tensor grid.
        - weights : Cell array of weights for each tensor grid.
        - size : Cell array of sizes for each tensor grid.
        - knots_per_dim : Cell array of knots per dimension for each tensor grid.
        - m : Cell array of level indices for each tensor grid.
        - coeff : Array of combination technique coefficients.
        - idx : Array of multi-indices corresponding to each tensor grid.
    C : ndarray
        Array of multi-indices used in the construction (0-based indexing).
    """
    if base==1:
        C = matlab_to_python_index(C)
    # we also build the set of S2 if provided
    if S2 is not None and len(S2) > 0:
        # get index set of S2. Note that if S2 has empty fields, C2==[]  ## TODO: check C2==[] when empty fields
        C2 = np.array(S2.idx)
    else:
        C2 = None
    # now compute the tensor grids out of the delta operators listed in C.
    # Exploit partial ordering of the sequence of multiindices
    # -----------------------------------------------
    # each multiindex of C is a delta grid. Given say [i1 i2 i3] i will get
    # (i1 - i1-1) x (i2 - i2-1) x (i3 - i3-1)
    # that is the grids associated to
    # [i1 i2 i3],[i1-1 i2 13],[i1 i2-1 i3],[i1-1 i2-1 i3] and so on
    # note that all of these multiindeces are also included in the initial set C
    # (if [i1 i2 i3] ratisfies the rule, all other will, because their indeces are equal or lower)
    # So the set of all possible grids (non delta grids) is the same set C, but some of them will cancel out
    # C is partially ordered (lexicographis): [x x x x i], are listed increasing with i,
    # [x x x j i] are listed increasing first with j and then with i ...
    # Now take a row of C, c. Because of the ordering, if you take c as a grid index
    # the same grid can appear again only from delta grids coming from rows following.
    # Now we scroll these rows following (say c2) and compute how many times c will be generated, and with
    # what sign. It has to be that d=c2-c has only 0 or 1
    # if this is the case, the sign of this new appearence of c could be both + or - .
    # To determine the sign, start with + and switch it every time a 1 appears in d
    # d=[0 1 0 0 1] => sign= +
    # the formula is (-1)^sum(d) ( d with even appearences of 1 gives a + , odd gives a -)
    # and just sum all the coefficients to see if c will survive or cancel out
    nn = C.shape[0]
    coeff = np.ones(nn)
    # I can at least restrict the search to multiindices whose first component is c(i) + 2, so I define
    __, bookmarks = matlab.unique(C[:, 0], 'first')
    bk = np.append(bookmarks[2:], [nn, nn])  # bookmarks - 1 not required
    # i.e. those who begin with 1 end at bookmark(3)-1, those who begin with 2-1 end at bookmark(4) and so on,
    # until there's no multiindex with c(i)+2
    for i in np.arange(nn):  # in np.arange(1, nn + 1).reshape(-1):
        cc = C[i, :]
        # recover the range in which we have to look. Observe that the first column of C contains necessarily 1,2,3 ...
        # so we can use them to access bk
        range_ = bk[cc[0]]
        for j in np.arange(i + 1, range_):  #.reshape(-1)
            # scroll c2, the following rows
            d = C[j, :] - cc
            if np.max(d) <= 1 and np.min(d) >= 0:
                # so if max(d)<=1 is false the other condition is not even checked
                coeff[i] = coeff[i] + (-1) ** np.sum(d)
    # now we can store only those grids who survived, i.e. coeff~=0
    # ------------------------------------------------------
    nb_grids = np.sum(coeff != 0)
    empty_cells = matlab.cell((1,nb_grids))
    S = matlab.struct('knots', empty_cells, 'weights', empty_cells, 'size', empty_cells, 'knots_per_dim', empty_cells,
               'm', empty_cells)
    fieldnms = matlab.fieldnames(S)
    coeff_condensed = np.zeros(nb_grids)
    ss = 0
    # for each nonzero coeff, generate the tensor grid and store it. If possible, recycle from S2.
    if C2 is not None and len(C2) > 0:
        logger_.debug('build sparse grid with tensor grid recycling')
        for j in np.arange(nn):
            if coeff[j] != 0:
                i = C[j, :]
                found, pos, _ = find_lexicographic(i, C2, 'nocheck')
                if found:
                    # disp('found')
                    # Note that at this point elements of S are tensor grids while S2 is a sparse grid therefore it has additional fields
                    # (coeff, idx). We thus need to copy field by field otherwise we'll have "assignment between dissimilar
                    # structures" error. We use dynamic filed names to this end
                    for fn in fieldnms:
                        setattr(S[ss], fn, getattr(S2[pos], fn))
                    # however we need to fix the weights. Indeed, they are stored in S2 as weights*coeff, so we need to reverse
                    # that multiplication
                    S[ss].weights = S[ss].weights / S2[pos].coeff
                else:
                    m = apply_lev2knots(i, lev2knots, N)
                    S[ss] = tensor_grid(N, m, knots)
                S[ss].weights = S[ss].weights * coeff[j]
                coeff_condensed[ss] = coeff[j]
                ss = ss + 1
    else:
        for j in np.arange(nn):
            if coeff[j] != 0:
                i = C[j, :]
                m = apply_lev2knots(i, lev2knots, N)
                S[ss] = tensor_grid(N, m, knots)
                S[ss].weights = S[ss].weights * coeff[j]
                coeff_condensed[ss] = coeff[j]
                ss = ss + 1
    # now store the coeff value. It has to be stored after the first loop, becuase tensor_grid returns a grid
    # WITHOUT coeff field, and Matlab would throw an error (Subscripted assignment between dissimilar structures)
    for ss in range(nb_grids):  # in np.arange(1, nb_grids + 1).reshape(-1):
        S[ss].coeff = coeff_condensed[ss]
    # similarly for the multiidx generating each tensor grid
    ss = 0
    for j in range(nn):
        if coeff[j] != 0:
            S[ss].idx = C[j, :]
            ss = ss + 1

    return S, C
