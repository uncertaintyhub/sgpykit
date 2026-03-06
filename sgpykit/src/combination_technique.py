import numpy as np

from sgpykit.util import matlab


def combination_technique(I):
    """
    Compute the coefficients of the combination technique expression of a sparse grid.

    The combination technique computes the coefficients for expressing a sparse grid as a
    linear combination of tensor grids. This function takes a multi-index set I (each row
    representing a multi-index associated with a Delta operator in the sparse grid
    construction) and computes the corresponding combination technique coefficients.

    Parameters
    ----------
    I : ndarray
        0-based multi-index set matrix (one row per index), must be admissible and
        lexicographically sorted. These properties are not checked by the function.

    Returns
    -------
    coeff : ndarray
        Vector containing the combination technique coefficients.

    Notes
    -----
    The input matrix I must be admissible and lexicographically sorted. The function does
    not verify these properties, so incorrect input may lead to unexpected results.
    """
    nn = I.shape[0]
    coeff = np.ones(nn)                     # initialize coefficients to 1: all c survive

    # I can at least restrict the search to multiindices whose first component is c(i) + 2, so I define
    __, bookmarks = matlab.unique(I[:, 0], 'first')
    max_val = int(I[:, 0].max())
    bk = np.empty(max_val+1, dtype=int)
    for v in range(0, max_val + 1):
        if v + 2 < max_val:
            bk[v] = bookmarks[v + 2]
        else:
            bk[v] = nn
    # i.e. those who begin with 1 end at bookmark(3)-1, those who begin with 2-1 end at bookmark(4) and so on,
    # until there's no multiindex with c(i)+2
    #
    # an old piece of code I still want to have written somewhere
    # range = i+ find( C(i+1:end,1)==cc(1)+2, 1 ) - 1;

    for i in range(nn):                     # scroll c
        cc = I[i, :]
        # recover the range in which we have to look. Observe that the first column of C contains necessarily 1,2,3 ...
        # so we can use them to access bk
        range_end = bk[cc[0]]
        for j in range(i + 1, range_end):
            # scroll c2, the following rows
            d = I[j, :] - cc
            if d.max() <= 1 and d.min() >= 0:   # much faster to check then if only 0 and 1 appears. Also, && is short-circuited,
                # so if max(d)<=1 is false the other condition is not even checked
                coeff[i] = coeff[i] + (-1) ** int(d.sum())
    return coeff
