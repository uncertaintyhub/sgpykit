import numpy as np


def delta_to_combitec(ii):
    """
    Compute the combination technique contributions of a multi-index.

    DELTA_TO_COMBITEC computes the combination technique contributions of a multi-index
    I = delta_to_combitec(ii)
    takes as input a multi-index (row-vector) ii that describes a delta operator in the sparse grids construction
    and returns the set I of the corresponding quadrature/interpolation operators (lexicosorted). It does *not*
    return instead the coefficient of the index in the combination technique, whose value is trivial to compute, see below.
    In formulas,
    Delta^ii = \\prod_{n=1}^N ( U_n^{ii_n} - U_n^{ii_n-1})
    where U_n^{i_n} is the quadrature/interpolation operator along direction n at level i_n.
    The Delta^ii can be expressed as linear combination of tensorized U_n operators:
    Delta^ii = \\sum_{kk \\in K} c_kk \\prod_n U_n^{kk_n}
    where c_kk = (-1)^sum(ii - kk);
    and I = sortrows(K),  while c_kk is not returned
    For instance
    Delta^[1 2] = (U_1^1 -U_1^0) * (U_2^2 -U_2^1) = U_1^1 * U_2^2 - U_1^1 * U_2^1 - U_1^0 * U_2^2 + U_1^0 * U_2^1
    and delta_to_combitec([1, 2]):
    ```
         0     1
         0     2
         1     1
         1     2
    ```
    Of course whenever ii_n = 0 for some n,  we should not take the difference ( U_n^{ii_n} - U_n^{ii_n-1}); delta_to_combitec does this too:
    delta_to_combitec([1, 1, 2]):
    ```
         0     0     1
         0     0     2
         0     1     1
         0     1     2
         1     0     1
         1     0     2
         1     1     1
         1     1     2
    ```
    but delta_to_combitec([1, 0, 2]):
    ```
         0     0     1
         0     0     2
         1     0     1
         1     0     2
    ```

    Parameters
    ----------
    ii : array_like
        Multi-index (row-vector) describing a delta operator in the sparse grids construction (0-based indexing).

    Returns
    -------
    I : ndarray
        Set of corresponding quadrature/interpolation operators (lexicosorted).
    """
    ii = np.atleast_1d(ii).flatten()
    N = len(ii)

    # I need to skip the components of ii equal to 0
    N_eff = np.sum(ii > 0)  # other implementation have similar runtimes, like length(find(ii>0)) or sum(ii~=0)

    # preallocate I
    I = np.zeros((2 ** N_eff, N), dtype=np.int64)

    k_eff = 1

    # we work column-by-columns, i.e. loop over N
    for k in range(N):
        # if the k-th component is not 0, replace the column of I, otherwise just leave the zeros. The columns is obtained by repeating T times
        # the basic vector, which contains C copies of ii(k) and C copies of ii(k)-1.
        if ii[k] != 0:
            times = 2 ** (k_eff - 1)
            copies = 2 ** (N_eff - k_eff)  # 2 x copies x times = #rows => 2 x copies x 2^(k_eff - 1)  = 2^N_eff => copies = 2^N_eff  / 2^k_eff

            row = 0

            for _ in range(times):
                I[row:row + copies, k] = ii[k] - 1  # <--- in this way, it's already lexicosrted.
                I[row + copies:row + 2 * copies, k] = ii[k]
                row = row + 2 * copies

            k_eff = k_eff + 1

    return I
