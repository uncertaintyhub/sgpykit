import numpy as np

from sgpykit.src.compute_modal_tensor import compute_modal_tensor
from sgpykit.util import matlab
from sgpykit.util.struct_array import StructArray


def convert_to_modal(S, Sr, nodal_values, domain, flags):
    """
    Convert a sparse grid interpolant to a modal (spectral) expansion.

    This function recasts a sparse grid interpolant as a sum of orthogonal polynomials,
    i.e., computes the spectral expansion of the interpolant.

    Parameters
    ----------
    S : array_like
        Sparse grid structure.
    Sr : array_like
        Reduced sparse grid structure.
    nodal_values : array_like
        Values of the target function evaluated on the reduced sparse grid.
        If the function is vector-valued (F: R^N -> R^V), then nodal_values
        is a matrix of size V x M, where M is the number of points in Sr.knots.
        If the function is scalar-valued, nodal_values is a row vector.
    domain : array_like
        Domain specification for the sparse grid. The format depends on the
        polynomial family:
        - For 'legendre', 'chebyshev': 2xN matrix [a1, a2, ...; b1, b2, ...]
          defining the hyper-rectangle bounds.
        - For 'hermite': 2xN matrix [mu1, mu2, ...; sigma1, sigma2, ...]
          defining normal distribution parameters.
        - For 'laguerre': 1xN matrix [lambda1, lambda2, ...] defining
          exponential distribution parameters.
        - For 'generalized laguerre': 2xN matrix [alpha1, alpha2, ...; beta1, beta2, ...]
          defining Gamma distribution parameters.
        - For 'jacobi': 4xN matrix [alpha1, alpha2, ...; beta1, beta2, ...; a1, a2, ...; b1, b2, ...]
          defining Beta distribution parameters on intervals [a_n, b_n].
        - For mixed families: cell array of length N, where each cell contains the parameters
          for the corresponding polynomial family.
    flags : str or list of str
        Polynomial family or families to use for the expansion. Supported values are:
        'legendre', 'chebyshev', 'hermite', 'laguerre', 'generalized laguerre', 'jacobi'.
        For mixed families, provide a list of strings.

    Returns
    -------
    modal_coeffs : ndarray
        Matrix of modal coefficients. Each row corresponds to a multi-index and has V components.
    K : ndarray
        Matrix of multi-indices, one per row.

    Notes
    -----
    The function processes each tensor grid in the sparse grid, computes the modal
    coefficients for each grid, and aggregates the coefficients by summing those with
    identical multi-indices.
    """
    # CONVERT_TO_MODAL recasts a sparse grid interpolant as a sum of orthogonal polynomials
    # i.e. computes the spectral expansion of the interpolant.
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'legendre') returns the Legendre expansion
    #       of the sparse grid interpolant. S is a sparse grid, SR is its reduced counterpart, NODAL_VALUES
    #       are the values of the target function (say F) evaluated on the reduced sparse grid, and has the same
    #       size of e.g. F_EVAL, output of EVALUATE_ON_SPARSE_GRID.

    #       More precisely, if F: R^N -> R^V and SR.KNOTS contains M points in R^N, then
    #       SR.KNOTS is a matrix NxM and NODAL_VALUES is a matrix VxM, i.e.,
    #       evaluations of F on different points of SR must be stored in NODAL_VALUES as columns.
    #       If F is a scalar-valued function then NODAL_VALUES is a row-vector

    #       DOMAIN is a 2xN matrix = [a1, a2, a3, ...; b1, b2, b3, ...] defining the lower and upper bound
    #       of the hyper-rectangle on which the sparse grid is defined
    #       The function returns the Legendre expansion as a matrix of coefficients MODAL_COEFFS, where
    #       the rows of MODAL_COEFF have V components (V columns), i.e. they are the "function coefficient" of the expansion
    #       of F: R^N -> R^V. K is a matrix containing the associated multi-indices, one per row.
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'chebyshev') returns the Chebyshev expansion
    #        of the sparse grid interpolant. See above for inputs and outputs.
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'hermite') returns the Hermite expansion
    #       of the sparse grid interpolant. Here, DOMAIN is a 2XN matrix = [mu1, mu2, mu3, ...; sigma1, sigma2, sigma3,...]
    #       i.e. the n-th variable of the sparse grid space has normal distribution with mean mu_n and std sigma_n
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'laguerre') returns the Laguerre expansion
    #       of the sparse grid interpolant. Here, DOMAIN is a 1XN matrix = [lambda1, lambda2, lambda3, ...]
    #       i.e. the n-th variable of the sparse grid space has exponential distribution with parameter lambda_n
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'generalized laguerre') returns the generalized Laguerre expansion
    #       of the sparse grid interpolant. Here, DOMAIN is a 2XN matrix = [alpha1, alpha2, alpha3, ...; beta1, beta2, beta3, ...]
    #       i.e. the n-th variable of the sparse grid space has Gamma distribution with parameters alpha_n and beta_n
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,'jacobi') returns the Jacobi expansion
    #       of the sparse grid interpolant. Here, DOMAIN is a 4XN matrix = [alpha1, alpha2, alpha3, ...; beta1, beta2, beta3, ...; a1, a2, a3, ...; b1, b2, b3, ...]
    #       i.e. the n-th variable of the sparse grid space has Beta distribution with parameters alpha_n and beta_n on the interval [a_n,b_n]
    # [MODAL_COEFFS,K] = CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,{<family1>,<family2>,<family3>,...}) returns the expansion
    #       of the sparse grid interpolant over polynomial of "mixed" type, according to the families specified in
    #       the last argument. For example:
    #       CONVERT_TO_MODAL(S,SR,NODAL_VALUES,DOMAIN,{'legendre','hermite','laguerre','jacobi','legendre'})
    #       converts the sparse grid interpolant in a sum of multi-variate polynomials that are products of
    #       univariate Legendre polynomials (directions 1 and 5), Hermite (direction 2), Lagurre (direction 3) and Jacobi (direction 4).
    #       Here DOMAIN is a cell array of lenght N where each cell contains the vector of the parameters (or a scalar value for the case 'laguerre')
    #       of the n-th family of polynomials.
    #       E.g. in the case above
    #       DOMAIN = {[a1;b1], [mu1;sigma1], lambda, [alpha2;beta2;a2;b2], [a3;b3]}

    if isinstance(flags, str):
        flags = [flags]
    if not all(flag in {'legendre', 'chebyshev', 'hermite', 'laguerre', 'generalized laguerre', 'jacobi'} for flag in
               flags):
        raise ValueError('One or more strings in FLAGS unrecognized.')

    if matlab.iscell(flags) and not matlab.iscell(domain):
        raise ValueError('Input argument DOMAIN must be a cell array. Each cell contains the domain for the corresponding polynomial.')

    # N is be the number of random variables
    N = Sr.knots.shape[0]
    # V is the output dimensionality, F:R^N -> R^V
    nodal_values = np.atleast_2d(nodal_values)
    V = nodal_values.shape[0]
    # one tensor grid at a time, compute the modal equivalent, then sum up
    # how many grid to process?
    nb_tensor_grids = len(S)
    # each tensor grid will result in vector of coefficients, u, that
    # are the coefficients of the modal (Legendre,Hermite) expansion of the
    # lagrangian interpolant. To combine these coefficients properly, I need
    # to store not only the coefficients u themselves, but also the associated
    # multi-indices. I store this information into a structure array, U,
    # whose lenght is nb_tensor_grids, with fields U(i).modal_coeffs, U(i).multi_indices, U(i).size (i.e.
    # how many polynomials are associated to each grid)

    # I will need this counter
    values_counter = 0
    # I also do a trick to preallocate U
    # TODO: check
    U = StructArray(nb_tensor_grids)
    # U[nb_tensor_grids-1].multi_indices = []
    # U[nb_tensor_grids-1].size = []
    # U[nb_tensor_grids-1].modal_coeffs = []
    for g in range(nb_tensor_grids):
        # some of the grids in S structure are empty, so I skip them
        if S[g].knots is None or len(S[g].knots) == 0:
            continue
        # recover the values of f on the current grid.
        # To do this, I need to use Sr.n, that is the mapping from [S.knots] to Sr.knots
        # To locate the knots of the g-th grid in the mapping Sr.n, i use values_counter
        grid_knots = slice(values_counter, values_counter + S[g]['size'])
        # extract the values of the g-th grid. If nodal_values is vector-valued, because F:R^N -> R^V
        # then the evaluation of each component of F, F_1, F_2, ... F_V on each point of the tensor
        # grid must be in a column vector, therefore I need to transpose nodal_values
        S_values = np.transpose(nodal_values[:, Sr.n[grid_knots]])
        # then update values_counter
        values_counter = values_counter + S[g].size
        # and compute modal
        U[g] = compute_modal_tensor(S[g], S_values, domain, flags)

    # now I need to put together all the U.multi_indices, summing the coefficients that
    # share the same multi_index. Same procedure as reduce sparse grid

    # First I build a list of all multi_indices and coefficients
    # I preallocate them to speed up code

    tot_midx = sum(U[g]['size'] for g in range(nb_tensor_grids))
    # this is the container of multi_indices
    All = np.zeros((tot_midx, N))
    # this is the container of coefficients
    all_coeffs = np.zeros((tot_midx, V))
    # this is the index that scrolls them
    l = 0
    # fill All and all_coeffs
    for g in range(nb_tensor_grids):
        if S[g].knots is None or len(S[g].knots) == 0:
            continue
        All[l:l + U[g]['size'], :] = U[g]['multi_indices']
        all_coeffs[l:l + U[g]['size'], :] = U[g]['modal_coeffs'] * S[g]['coeff']
        # a row vector (row of all_coeffs) and on the right we have
        # a column vector
        l = l + U[g].size

    # get rid of identical ones, by sorting and taking differences between cosecutive rows
    # I need to store the sorter vector, to sum up all coefficients with
    # the same multi_index

    All_sorted, sorter = matlab.sortrows(All)
    # taking differences between consecutive rows, I have a way
    # to build the unique version of All. Remember diff has 1 row
    # less, so I have to add a ficticious row at the end of All

    dAll_sorted = np.diff(np.vstack((All_sorted, All_sorted[-1] + 1)), axis=0)
    # if a row of dAll_sorted is made of 0s then I have
    # to discard the corresponding row of All

    selector = np.any(dAll_sorted != 0, axis=1)

    # store uniqued version of All
    K = All_sorted[selector]
    # now I have to sum all the coefficients.

    # sort coefficients in the same way as multi_indices
    all_coeffs_sorted = all_coeffs[sorter]
    # initialize modal_coeffs, that is the final output
    nb_modal_coeffs = K.shape[0]
    modal_coeffs = np.zeros((nb_modal_coeffs, V))
    # compute them one at a time

    l = 0  # scrolls all_coeffs_sorted
    for k in range(nb_modal_coeffs):
        ss = all_coeffs_sorted[l]
        while not selector[l]:
            l += 1
            ss += all_coeffs_sorted[l]
        modal_coeffs[k] = ss
        l += 1
    return modal_coeffs, K
