import numpy as np

from sgpykit.tools.idxset_functions.multiidx_box_set import multiidx_box_set
from sgpykit.tools.polynomials_functions.cheb_eval import cheb_eval
from sgpykit.tools.polynomials_functions.cheb_eval_multidim import cheb_eval_multidim
from sgpykit.tools.polynomials_functions.generalized_lagu_eval_multidim import generalized_lagu_eval_multidim
from sgpykit.tools.polynomials_functions.herm_eval import herm_eval
from sgpykit.tools.polynomials_functions.herm_eval_multidim import herm_eval_multidim
from sgpykit.tools.polynomials_functions.jacobi_prob_eval_multidim import jacobi_prob_eval_multidim
from sgpykit.tools.polynomials_functions.lagu_eval_multidim import lagu_eval_multidim
from sgpykit.tools.polynomials_functions.lege_eval import lege_eval
from sgpykit.tools.polynomials_functions.lege_eval_multidim import lege_eval_multidim
from sgpykit.util import matlab
from sgpykit.util.matlab import struct


def compute_modal_tensor(S, S_values, domain, flags):
    """
    Convert a tensor grid interpolant to a modal expansion in orthogonal polynomials.

    This function takes a tensor grid and its associated point evaluations, and converts
    the resulting Lagrange multivariate interpolant to a sum of orthogonal polynomials.
    The type of polynomials used can be specified via the `flags` parameter.

    Parameters
    ----------
    S : struct
        A struct containing the tensor grid information, including knots and size.
    S_values : ndarray
        The values of the interpolant at the tensor grid points.
    domain : ndarray or list of ndarrays
        The domain of the sparse grid. The format depends on the polynomial family:
        - For Legendre, Chebyshev: 2xN matrix [a1, a2, ...; b1, b2, ...]
        - For Hermite: 2xN matrix [mu1, mu2, ...; sigma1, sigma2, ...]
        - For Laguerre: 1xN matrix [lambda1, lambda2, ...]
        - For generalized Laguerre: 2xN matrix [alpha1, alpha2, ...; beta1, beta2, ...]
        - For Jacobi: 4xN matrix [alpha1, alpha2, ...; beta1, beta2, ...; a1, a2, ...; b1, b2, ...]
        - For mixed families: a cell array where each cell contains the domain for the corresponding polynomial.
    flags : str or list of str
        The type of orthogonal polynomials to use. Can be a single string or a list of strings
        for mixed families. Supported values are:
        - 'legendre'
        - 'chebyshev'
        - 'hermite'
        - 'laguerre'
        - 'generalized laguerre'
        - 'jacobi'

    Returns
    -------
    U : struct
        A struct with the following fields:
        - size: the number of polynomials in the expansion.
        - multi_indices: the multi-indices corresponding to each polynomial.
        - modal_coeffs: the coefficients of the modal expansion.

    Raises
    ------
    ValueError
        If one or more strings in `flags` are unrecognized, or if the input argument `domain`
        is not a cell array when `flags` is a cell array.
    """
    # COMPUTE_MODAL_TENSOR given a tensor grid and the values on it, re-express the interpolant
    # as a modal expansion.
    # U=COMPUTE_MODAL_TENSOR(S,S_VALUES,DOMAIN,'legendre') considers the tensor grid S on the
    #       hyper-rectangle DOMAIN with associated point evaluations S_VALUES and converts the
    #       resulting lagrangian multivariate interpolant to a sum of Legendre polynomials.
    #       DOMAIN is a 2xN matrix = [a1, a2, a3, ...; b1, b2, b3, ...]
    #       defining the hyper-rectangluar domain of the sparse grid: (a1,b1) x (a2,b2) x ...
    #       U is a struct with fields U.size (the number of Legendre polynomials needed),
    #       U.multi_indices (one multi-index per Legendre polynomial), U.coeffs
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,'chebyshev') works as the previous call, using
    #       Chebyshev polynomials
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,'hermite') works as the previous call, using
    #       Hermite polynomials. Here DOMAIN is a 2XN matrix = [mu1, mu2, mu3, ...; sigma1, sigma2, sigma3,...]
    #       such that the first variable has normal distribution with mean mu1 and std sigma1
    #       and so on.
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,'laguerre') works as the previous call, using
    #       Laguerre polynomials. Here DOMAIN is a 1XN matrix = [lambda1, lambda2, lambda3, ...]
    #       such that the first variable has exponential distribution with parameter lambda1 and so on.
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,'generalized laguerre') works as the previous call, using
    #       generalized Laguerre polynomials. Here DOMAIN is a 2XN matrix = [alpha1, alpha2, alpha3, ...; beta1, beta2, beta3, ...]
    #       such that the first variable has Gamma distribution with parameters alpha1 and beta1 and so on.
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,'jacobi') works as the previous call, using
    #       Jacobi polynomials. Here DOMAIN is a 4XN matrix = [alpha1, alpha2, alpha3, ...; beta1, beta2, beta3, ...; a1, a2, a3, ...; b1, b2, b3, ...]
    #       such that the first variable has Beta distribution with parameters alpha1 and beta1 on the interval [a1,b1] and so on.
    # U=COMPUTE_MODAL_TENSOR(S,S_values,domain,{<family1>,<family2>,<family3>,...}) works as the previous call, using
    #       polynomials of type <family-n> in direction n. Here DOMAIN is a cell array of lenght N
    #       where each cell contains the matrix giving the parameters of the n-th family of polynomials.
    #       For example:
    #       COMPUTE_MODAL_TENSOR(S,S_VALUES,DOMAIN,{'legendre','hermite','laguerre','jacobi','legendre'})
    #       with
    #       DOMAIN = {[a1;b1], [mu1;sigma1], lambda, [alpha2;beta2;a2;b2], [a3;b3]}

    if isinstance(flags, str):
        flags = [flags]
    if not all(flag in {'legendre', 'chebyshev', 'hermite', 'laguerre', 'generalized laguerre', 'jacobi'} for flag in
               flags):
        raise ValueError('One or more strings in FLAGS unrecognized.')

    if matlab.iscell(flags) and not matlab.iscell(domain):
        raise ValueError('Input argument DOMAIN must be a cell array. Each cell contains the domain for the corresponding polynomial.')

    # I will need the knots in each dimension separately.
    # As the number of knots is different in each direction, I use a cell array

    nb_dim = S.knots.shape[0]
    # knots_per_dim=cell((1,nb_grids));
    # for dim=1:nb_dim
    #     knots_per_dim{dim}=unique(S.knots[dim,:]);
    # end

    knots_per_dim = S.knots_per_dim
    # The modal expansion in i-th direction uses up to degree k, with k as follows

    degrees = np.zeros(nb_dim, dtype=np.int64)
    for dim in range(nb_dim):
        degrees[dim] = len(knots_per_dim[dim]) - 1

    # the modal polynomials to be used are s.t. the corresponding multi-indices have
    # all components less than or equal to the maximum degree

    I,_ = multiidx_box_set(degrees,0)
    # return multiindex_set as
    nb_multiindices = I.shape[0]
    # safety check. I have solve a system, so I need the vandermonde matrix to be squared!
    rows = S.size
    cols = nb_multiindices

    if rows != cols:
        raise ValueError('Vandermonde matrix is not square!')

    # now create the vandermonde matrix with evaluation of each multi-index in every point
    # of the grid

    V = np.zeros((rows, cols))
    if len(flags) == 1:
        flags = flags[0]
    if isinstance(flags, str):  # the second condition for when the function is called on one single family of polynomials
        for c in range(cols):
            k = I[c, :]
            # vc = lege_eval_multidim(interval_map(S['knots']), k, domain[0], domain[1])
            if flags == 'legendre':
                vc = lege_eval_multidim(S['knots'], k, domain[0], domain[1])
            elif flags == 'hermite':
                vc = herm_eval_multidim(S['knots'], k, domain[0], domain[1])
            elif flags == 'chebyshev':
                vc = cheb_eval_multidim(S['knots'], k, domain[0], domain[1])
            elif flags == 'laguerre':
                vc = lagu_eval_multidim(S['knots'], k, domain)
            elif flags == 'generalized laguerre':
                vc = generalized_lagu_eval_multidim(S['knots'], k, domain[0], domain[1])
            elif flags == 'jacobi':
                vc = jacobi_prob_eval_multidim(S['knots'], k, domain[0], domain[1], domain[2], domain[3])
            else:
                raise ValueError('Unknown family of polynomials')
            V[:, c] = vc.T
    else:
        for c in range(cols):
            k = I[c, :]
            vc = np.ones(S['knots'][0].shape)
            for n in range(nb_dim):
                if flags[n] == 'legendre':
                    # vc = vc * lege_eval(S['knots'][n], k[n], domain[0][n], domain[1][n])
                    vc = vc * lege_eval(S['knots'][n], k[n], domain[n][0], domain[n][1])
                elif flags[n] == 'hermite':
                    # vc = vc * herm_eval(S['knots'][n], k[n], domain[0][n], domain[1][n])
                    vc = vc * herm_eval(S['knots'][n], k[n], domain[n][0], domain[n][1])
                elif flags[n] == 'chebyshev':
                    # vc = vc * cheb_eval(S['knots'][n], k[n], domain[0][n], domain[1][n])
                    vc = vc * cheb_eval(S['knots'][n], k[n], domain[n][0], domain[n][1])
                elif flags[n] == 'laguerre':
                    vc = vc * lagu_eval_multidim(S['knots'][n], k[n], domain[n])
                elif flags[n] == 'generalized laguerre':
                    vc = vc * generalized_lagu_eval_multidim(S['knots'][n], k[n], domain[n][0], domain[n][1])
                elif flags[n] == 'jacobi':
                    vc = vc * jacobi_prob_eval_multidim(S['knots'][n], k[n], domain[n][0], domain[n][1], domain[n][2],
                                                        domain[n][3])
                else:
                    raise ValueError('Unknown family of polynomials')
            V[:, c] = vc.T

    # now solve the system

    sol = np.linalg.solve(V, S_values)
    U = struct(multi_indices=I, size=nb_multiindices, modal_coeffs=sol)

    return U
