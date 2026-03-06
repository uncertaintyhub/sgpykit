import os

import numpy as np
import scipy

from sgpykit.tools.knots_functions.knots_gamma import knots_gamma
from sgpykit.tools.polynomials_functions.lagr_eval import lagr_eval
from sgpykit.util import matlab


def knots_gamma_leja(n, alpha, beta, saving_flag=None):
    """
    Compute the first n collocation points and corresponding weights for the weighted Leja sequence for integration with respect to the Gamma density.

    This function returns the first n collocation points (x) and the corresponding weights (w),
    for the weighted Leja sequence for integration with respect to the weight function
    rho(x) = beta^(alpha+1)/Gamma(alpha+1) * x^alpha * exp(-beta*x),
    i.e., the density of a standard Gamma random variable with range [0, +inf), alpha > -1, beta > 0.

    Knots and weights are computed following the work:
    A. Narayan, J. Jakeman, "Adaptive Leja sparse grid constructions for stochastic collocation and high-dimensional approximation"
    SIAM Journal on Scientific Computing, Vol. 36, No. 6, pp. A2952--A2983, 2014.

    The knots are computed over the interval [0, 300]. This choice of interval is compatible with -1 < alpha <= 30.
    If alpha > 30, a larger interval is required and an error is raised.
    The maximum number of points that can be computed is 50, for compatibility with the interval [0, 300].

    If <saving_flag> is set to 'on_file', the function will compute 50 knots and weights, store them in a MATLAB style .mat file
    in the local folder and load results at next calls with the same values of alpha. Note that different values of beta
    act only as a rescaling factor, therefore the calls
    knots_gamma_leja(n, alpha, beta1, 'on_file')
    knots_gamma_leja(n, alpha, beta2, 'on_file')
    can use the same precomputed file.
    If no saving flag is provided, i.e., the function is called as knots_gamma_leja(n, alpha, beta),
    computations of nodes and weights will not be saved. This might become time-consuming for large sparse grids
    (the computation of the Leja knots is done by brute-force optimization).

    Parameters
    ----------
    n : int
        Number of collocation points to compute.
    alpha : float
        Shape parameter of the Gamma distribution (alpha > -1).
    beta : float
        Rate parameter of the Gamma distribution (beta > 0).
    saving_flag : str, optional
        Flag to enable saving and loading of precomputed knots and weights. If 'on_file', the function will save and load from a .mat file.

    Returns
    -------
    x : ndarray
        Array of collocation points.
    w : ndarray
        Array of corresponding weights.

    Raises
    ------
    AssertionError
        If n <= 0.
    NotImplementedError
        If alpha > 30.
    Exception
        If n > 50.
    ValueError
        If saving_flag is not 'on_file'.
    """
    assert n > 0
    if alpha > 30:
        raise NotImplementedError(f'alpha={alpha} is too large. Select alpha<=30')

    if n > 50:
        raise Exception(f'OutOfTable: this number of points is not available: {n}')

    filename = None
    # TODO: test and benchmark the cache-on-disk feature
    if saving_flag is not None:
        if 'on_file' == saving_flag:
            save_and_load = True
            filename = f'SGK_knots_weights_gamma_leja_file_alpha_{alpha}.mat'
        else:
            raise ValueError('unknown saving option for knots_gamma_leja')
    else:
        save_and_load = False

    if save_and_load:
        if os.path.exists(str(filename)):
            # load from file
            mat_contents = scipy.io.loadmat(filename)
            Wall = mat_contents['Wall']
            Xall = mat_contents['Xall']
            if n > len(Xall):
                # but is good to keep it in case we relax the n=50 limit
                raise Exception(f'OutOfTable: not enough gamma points computed on file. Remove files {filename} and run again KNOTS_GAMMA_LEJA with "on_file" option and larger n')
            X = Xall[:n]
            W = Wall[:n, n-1]
        else:
            # compute 50 nodes
            Xall, Wall = get_GammaLejaKnotsAndWeights(50, alpha, 'all')
            scipy.io.savemat(filename, {'Xall': Xall, 'Wall': Wall})
            X = Xall[:n]
            W = Wall[:n, n-1]
    else:
        # just compute
        X, W = get_GammaLejaKnotsAndWeights(n, alpha, 'current')

    # modifies points according to beta
    X = X / beta
    # sort knots increasingly and weights accordingly. Weights need to be row vectors
    x, sorter = matlab.sort(X)
    w = W[sorter]
    return x, w


def get_GammaLejaKnotsAndWeights(n, alpha, which_weights):
    """
    Compute the Gamma Leja knots and corresponding weights.

    This function computes the Gamma Leja knots and corresponding weights for integration with respect to the Gamma density.
    The knots are computed over the interval [0, 300] using an adaptive brute-force optimization.

    Parameters
    ----------
    n : int
        Number of collocation points to compute.
    alpha : float
        Shape parameter of the Gamma distribution (alpha > -1).
    which_weights : str
        Flag to determine the type of weights to compute. If 'all', compute weights for all intermediate rule sizes. If 'current', compute weights only for the current rule size.

    Returns
    -------
    X : ndarray
        Array of collocation points.
    W : ndarray
        Array of corresponding weights.

    Raises
    ------
    ValueError
        If which_weights is not 'all' or 'current'.
    """
    # Tolerance for node computation
    tol = 1e-13
    # Initial and refinement step size
    h0 = 1e-2
    # Initial or coarse search grid
    x = np.arange(0, 300 + h0, h0)
    # Corresponding weight function values
    w = x**(0.5 * alpha) * np.exp(-0.5 * x)
    # Initializing node vector
    X = np.zeros(n)
    X[0] = alpha + 1

    # Number of points to be computed
    NMAX = n

    # Loop over length of node vector
    for j in range(1, NMAX):
        # Update weighted node function to be maximized
        x_k = X[j-1]
        w = np.abs(x - x_k) * w

        # Search for maximum on coarse level
        ind = np.argmax(w)

        # Adaptively refine search grid around maximum on current grid
        h = h0
        x_fine = x

        while h > tol and len(x_fine) > 1:
            h = h0 * h  # refine step size
            if ind == 0:
                x_fine = np.arange(x_fine[ind], x_fine[ind+1] + h, h)
            else:
                x_fine = np.arange(x_fine[ind-1], x_fine[ind+1] + h, h)  # refine grid around maximum
            w_fine = x_fine**(0.5 * alpha) * np.exp(-0.5 * x_fine)  # compute weights on finer grid
            # compute node function values on finer grid
            w_fine = np.abs(np.prod(np.repeat(x_fine[:, np.newaxis], j, axis=1) - np.repeat(X[:j, np.newaxis].T, len(x_fine), axis=0), axis=1)) * w_fine

            # Search for maximum on finer grid
            ind = np.argmax(w_fine)

        # Update node vector
        X[j] = x_fine[ind]

    # Computation of corresponding weights
    if which_weights == 'all':
        W = np.zeros((NMAX, NMAX)) # we store quadrature weights for rule with P points as P-th column
        for n in range(1, NMAX + 1):
            nodes = X[:n]
            x_quad, w_quad = knots_gamma((n + 1) // 2, alpha, beta=1)
            mask = np.ones(n, dtype=bool)
            for k in range(n):
                mask[k] = False
                W[k, n - 1] = np.dot(lagr_eval(nodes[k], nodes[mask], x_quad), w_quad)  # TODO: compute all weights at once?
                mask[k] = True
    elif which_weights == 'current':
        W = np.zeros(NMAX)
        nodes = X
        x_quad, w_quad = knots_gamma((NMAX + 1) // 2, alpha, beta=1)
        mask = np.ones(NMAX, dtype=bool)
        for k in range(NMAX):
            mask[k] = False
            W[k] = np.dot(lagr_eval(nodes[k], nodes[mask], x_quad), w_quad)  # TODO: compute all weights at once?
            mask[k] = True
    else:
        raise ValueError("Invalid value for which_weights. Use 'all' or 'current'.")

    return X, W
