import os

import logging
import numpy as np
import scipy

from sgpykit.tools.knots_functions.knots_beta import knots_beta
from sgpykit.tools.polynomials_functions.lagr_eval import lagr_eval
from sgpykit.util import matlab

logger = logging.getLogger(__name__)


def knots_beta_leja(n, alpha, beta, x_a, x_b, type_, saving_flag=None):
    """
    Compute the first n collocation points (x) and the corresponding weights (w)
    for the weighted Leja sequence for integration w.r.t to the weight function
    rho(x)=Gamma(alpha+beta+2)/ (Gamma(alpha+1)*Gamma(beta+1)*(x_b-x_a)^(alpha+beta+1)) * (x-x_a)^alpha * (x_b-x)^beta
    i.e. the density of a Beta random variable with range [x_a,x_b], alpha,beta>-1.

    Knots and weights are computed following the work
    A. Narayan, J. Jakeman, "Adaptive Leja sparse grid constructions for stochastic collocation and high-dimensional approximation"
    SIAM Journal on Scientific Computing,  Vol. 36, No. 6, pp. A2952--A2983, 2014

    Parameters
    ----------
    n : int
        Number of collocation points.
    alpha : float
        Shape parameter of the Beta distribution.
    beta : float
        Shape parameter of the Beta distribution.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.
    type_ : str
        Type of Leja sequence ('line' or 'sym_line').
    saving_flag : str, optional
        Flag to save and load computed knots and weights ('on_file').

    Returns
    -------
    x : ndarray
        Collocation points.
    w : ndarray
        Corresponding weights.

    Notes
    -----
    Knots are sorted increasingly before returning (weights are returned in the corresponding order).
    if <saving_flag> is set to 'on_file', the function will compute max(n,50) knots and weights, store them in a MATLAB style .mat file
    in the local folder and load results at next calls with the same values of alpha and beta. Note that nodes on
    different intervals [x_a, x_b] can be obtained by linear translations from the reference interval [-1, 1],
    therefore calls like
    knots_beta_leja(n,alpha,beta,a1,b1,<type>,'on_file')
    knots_beta_leja(n,alpha,beta,a2,b2,<type>,'on_file')
    can use the same precomputed file.
    if no saving flag is provided, i.e., the function is called as knots_beta_leja(n,alpha,beta,x_a,x_b,<type>),
    computations of nodes and weights will not be saved. This might become time-consuming for large sparse grids
    (the computation of the Leja knots is done by brute-force optimization).
    Follows the description of the choices of <type>.
    -----------------------------------------------------------
    [X,W] = KNOTS_BETA_LEJA(n,alpha,beta,x_a,x_b,'line') given X(1)=(alpha+1)/(alpha+beta+2)
     recursively defines the n-th point by
      X_n= argmax_[x_a,x_b] prod_{k=1}^{n-1} abs(X-X_k)
    [X,W] = KNOTS_BETA_LEJA(N,mi,sigma,'sym_line') given X(1)=0 recursively
      defines the n-th and (n+1)-th point by
      X_n= argmax prod_{k=1}^{n-1} abs(X-X_k)
      X_(n+1) = symmetric point of X_n with respect to 0
    """
    assert n > 0
    # TODO: test and benchmark the cache-on-disk feature
    filename = None
    save_and_load = False
    if saving_flag is not None:
        if 'on_file' == saving_flag:
            save_and_load = True
            filename = f'SGK_knots_weights_beta_{type_}_leja_file_alpha_{alpha}_beta_{beta}.mat'
        else:
            raise ValueError('unknown saving option for knots_beta_leja')

    if 'line' == type_:
        if save_and_load:
            if os.path.exists(str(filename)):
                # load from file
                mat_contents = scipy.io.loadmat(filename)
                Wall = mat_contents['Wall']
                Xall = mat_contents['Xall']
                if n > len(Xall):
                    raise Exception(f'OutOfTable: not enough beta points computed on file. Remove file {filename} and run again KNOTS_BETA_LEJA with "on_file" option and larger n')
                X = Xall[:n]
                W = Wall[:n, n - 1]
            else:
                # compute a large number of nodes
                Xall, Wall = get_BetaLejaKnotsAndWeights(np.amax(50, n), alpha, beta, 'all')
                # save on file
                scipy.io.savemat(filename, {'Xall': Xall, 'Wall': Wall})
                X = Xall[:n]
                W = Wall[:n, n-1]
        else:
            # just compute what asked
            X, W = get_BetaLejaKnotsAndWeights(n, alpha, beta, 'current')
        # --------------------------------------------------
    else:
        if 'sym_line' == type_:
            if alpha != beta:
                logger.warning('InconsistentInput: The shape parameters alpha and beta are not equal. Hence, the beta pdf is not symmetric and working with symmetric knots is not recommended.')
            if save_and_load:
                if os.path.exists(str(filename)):
                    # load from file
                    mat_contents = scipy.io.loadmat(filename)
                    Wall = mat_contents['Wall']
                    Xall = mat_contents['Xall']
                    if n > len(Xall):
                        raise Exception(f'OutOfTable: not enough beta points computed on file. Remove files {filename} and run again KNOTS_BETA_LEJA with "on_file" option and larger n')
                    X = Xall[:n]
                    W = Wall[:n, n - 1]
                else:
                    # disp('compute 50 nodes')
                    Xall, Wall = get_SymBetaLejaKnotsAndWeights(np.amax(50, n), alpha, beta, 'all')
                    # save on file
                    scipy.io.savemat(filename, {'Xall': Xall, 'Wall': Wall})
                    X = Xall[:n]
                    W = Wall[:n, n - 1]
            else:
                # just compute what asked
                X, W = get_SymBetaLejaKnotsAndWeights(n, alpha, beta, 'current')
            # --------------------------------------------------
        else:
            raise ValueError('unknown Leja type')

    # modifies points according to x_a and x_b
    X = ((x_a + x_b) + (x_b - x_a) * X) / 2
    # sort knots increasingly and weights accordingly. Weights need to be row vectors
    x, sorter = matlab.sort(X)
    w = W[sorter]
    return x, w


def get_BetaLejaKnotsAndWeights(n, alpha, beta, which_weights):
    """
    Compute the Beta Leja knots and weights.

    Parameters
    ----------
    n : int
        Number of knots.
    alpha : float
        Shape parameter of the Beta distribution.
    beta : float
        Shape parameter of the Beta distribution.
    which_weights : str
        Flag to compute all weights or only the current weights ('all' or 'current').

    Returns
    -------
    X : ndarray
        Knots.
    W : ndarray
        Weights.
    """
    # Tolerance for node computation
    tol = 1e-16
    # Initial and refinement step size
    h0 = 1e-2
    # Initial or coarse search grid
    x = np.arange(-1, 1 + h0, h0)
    # Corresponding weight function values
    if alpha<0 and np.any(np.isclose(x,1.0)) or beta<0 and np.any(np.isclose(x,-1.0)):
        logger.warning("Some values will lead to inf/nan results.")
    w = (1 - x)**(0.5 * alpha) * (1 + x)**(0.5 * beta)
    # Initializing node vector
    X = np.zeros(n)
    X[0] = (alpha + 1) / (alpha + beta + 2)

    # number of points to be computed
    NMAX = n

    # Loop over length of node vector
    for j in range(1, NMAX):
        # Update weighted node function to be maximized
        x_k = X[j - 1]
        w = np.abs(x - x_k) * w

        # Search for maximum on coarse level
        ind = np.argmax(w)

        # Adaptively refine search grid around maximum on current grid
        h = h0
        x_fine = x
        while h > tol:
            h = h0 * h  # refine step size
            if ind == 0:
                x_fine = np.arange(x_fine[ind], x_fine[ind + 1] + h, h)
            elif ind == len(x_fine) - 1:
                x_fine = np.arange(x_fine[ind - 1], x_fine[ind] + h, h)
            else:
                x_fine = np.arange(x_fine[ind - 1], x_fine[ind + 1] + h, h)
            w_fine = (1 - x_fine)**(0.5 * alpha) * (1 + x_fine)**(0.5 * beta)  # compute weights on finer grid
            # compute node function values on finer grid
            w_fine = np.abs(np.prod(np.repeat(x_fine[:, np.newaxis], j, axis=1) - np.repeat(X[:j, np.newaxis].T, len(x_fine), axis=0), axis=1)) * w_fine
            # Search for maximum on finer grid
            ind = np.argmax(w_fine)

        # Update node vector
        X[j] = x_fine[ind]

    #-------------------------------------------------------
    # Computation of corresponding weights
    if which_weights == 'all':  # all weights for formulas with 1 node up to NMAX nodes
        W = np.zeros((NMAX, NMAX))  # we store quadrature weights for rule with P points as P-th column
        for n in range(1, NMAX + 1):
            nodes = X[:n]
            x_quad, w_quad = knots_beta(np.ceil(n / 2).astype(np.int64), alpha, beta, -1, 1)
            nnn = np.arange(1, n + 1)
            for k in nnn - 1:
                W[k, n - 1] = np.dot(lagr_eval(nodes[k], nodes[nnn != k + 1], x_quad), w_quad)
    elif which_weights == 'current':  # only weights for formula with NMAX nodes
        W = np.zeros(NMAX)
        nodes = X
        x_quad, w_quad = knots_beta(np.ceil(NMAX / 2).astype(np.int64), alpha, beta, -1, 1)
        nnn = np.arange(1, NMAX + 1)
        for k in nnn - 1:
            W[k] = np.dot(lagr_eval(nodes[k], nodes[nnn != k + 1], x_quad), w_quad)
    else:
        raise ValueError("Invalid value for which_weights. Use 'all' or 'current'.")

    return X, W


def get_SymBetaLejaKnotsAndWeights(n, alpha, beta, which_weights):
    """
    Compute the symmetric Beta Leja knots and weights.

    Parameters
    ----------
    n : int
        Number of knots.
    alpha : float
        Shape parameter of the Beta distribution.
    beta : float
        Shape parameter of the Beta distribution.
    which_weights : str
        Flag to compute all weights or only the current weights ('all' or 'current').

    Returns
    -------
    X : ndarray
        Knots.
    W : ndarray
        Weights.
    """
    # Tolerance for node computation
    tol = 1e-16
    # Initial and refinement step size
    h0 = 0.01
    # Initial or coarse search grid
    x = np.arange(-1, 1 + h0, h0)
    # Corresponding weight function values
    w = (1 - x) ** (0.5 * alpha) * (1 + x) ** (0.5 * beta)
    # Number of points to be computed
    NMAX = n
    # Initializing node vector
    X = np.zeros(NMAX)

    # Loop over length of node vector
    for j in range(1, NMAX):
        # Update weighted node function to be maximized
        x_k = X[j - 1]
        w = np.abs(x - x_k) * w
        if j % 2 == 1:
            # Symmetric node
            X[j] = -X[j - 1]
        else:
            # Search for maximum on coarse level
            ind = np.argmax(w)
            # Adaptively refine search grid around maximum on current grid
            h = h0
            x_fine = x
            while h > tol:
                h *= h0
                if ind == 0:
                    x_fine = np.arange(x_fine[ind], x_fine[ind + 1] + h, h)
                elif ind == len(x_fine) - 1:
                    x_fine = np.arange(x_fine[ind - 1], x_fine[ind] + h, h)
                else:
                    x_fine = np.arange(x_fine[ind - 1], x_fine[ind + 1] + h, h)
                w_fine = (1 - x_fine) ** (0.5 * alpha) * (1 + x_fine) ** (0.5 * beta)
                # Compute node function values on finer grid
                w_fine = np.abs(
                    np.prod(np.repeat(x_fine[:, None], j, axis=1) - np.tile(X[:j], (len(x_fine), 1)), axis=1)) * w_fine
                # Search for maximum on finer grid
                ind = np.argmax(w_fine)
            # Update node vector
            X[j] = x_fine[ind]

    # Computation of corresponding weights
    if which_weights == 'all':
        W = np.zeros((NMAX, NMAX))
        for n in range(NMAX):
            nodes = X[:n + 1]
            x_quad, w_quad = knots_beta(np.ceil(n / 2).astype(np.int64), alpha, beta, -1, 1)
            nnn = np.arange(n + 1)
            for k in nnn:
                W[k, n] = np.dot(lagr_eval(nodes[k], nodes[nnn != k], x_quad), w_quad)
    elif which_weights == 'current':
        W = np.zeros((NMAX, 1))
        nodes = X
        x_quad, w_quad = knots_beta(np.ceil(NMAX / 2).astype(np.int64), alpha, beta, -1, 1)
        nnn = np.arange(NMAX)
        for k in nnn:
            W[k] = np.dot(lagr_eval(nodes[k], nodes[nnn != k], x_quad), w_quad)
    else:
        raise ValueError("Invalid value for which_weights. Use 'all' or 'current'.")

    return X, W
