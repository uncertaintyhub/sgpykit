import numpy as np

from sgpykit.main.convert_to_modal import convert_to_modal
from sgpykit.util import matlab

def compute_sobol_indices_from_sparse_grid(S, Sr, nodal_values, domain, flags):
    """
    Compute Sobol indices from a sparse grid approximation of a scalar-valued function.

    This function computes the Sobol indices of a scalar-valued function f:R^N -> R in two steps:
    1) Converts the sparse grid approximation of that function into its equivalent Polynomial Chaos Expansion (PCE).
    2) Performs algebraic manipulations of the PCE coefficients.

    Parameters
    ----------
    S : dict
        Sparse grid structure.
    Sr : dict
        Reduced sparse grid structure.
    nodal_values : numpy.ndarray
        Nodal values of the function on the sparse grid.
    domain : dict
        Domain information.
    flags : dict
        Flags for the conversion to modal.

    Returns
    -------
    Sob_i : numpy.ndarray
        Principal Sobol indices.
    Tot_Sob_i : numpy.ndarray
        Total Sobol indices.
    Mean : float
        Expected value of the function.
    Var : float
        Variance of the function.
    """
    # first, call convert_to_modal
    gpce_coeffs, idx_set = convert_to_modal(S, Sr, nodal_values, domain, flags)
    N = Sr.knots.shape[0]
    Nb_coeff = len(gpce_coeffs)
    # I create two matrices that I use to mark the coefficients that I need to sum to get the Sobol indices.
    # Both have one column for each variable and one row for each gPCE coefficient.

    # In the first one, I mark with 1 the entry (i,j) if the i-th coefficient must be used to compute
    # the principal Sobol index of the j-th variable.

    # In the second one, I do the same but for the Total Sobol index
    Fact_STi = np.zeros((Nb_coeff,N))
    Fact_Si = np.zeros((Nb_coeff,N))
    # loop on coefficients, for each one decide if I should mark it or not. Note that the first PCE coeff
    # gives the mean of the function and does not enter the Sobol index computation, so I can start from
    # the 2nd coefficient

    for r in range(1,Nb_coeff):
        # check the multi-idx of this coefficient and find which entries are non-zero
        Ind = matlab.find(idx_set[r,:] > 0)
        # the current coefficients is then to be used to compute the total sobol index
        # of all those variables
        Fact_STi[r,Ind] = 1
        # moreover, if there is only one non-zero entry in the multi-idx, than this coefficient
        # must be marked for use in the computation of the principal sobol index
        if len(Ind) == 1:
            Fact_Si[r,Ind] = 1

    # compute mean and variance from the PCE coefficients
    Mean = gpce_coeffs[0][0]
    gpce_squared = (gpce_coeffs ** 2)
    Var = np.sum(gpce_squared) - Mean ** 2
    # now for each variable, sum all coefficients that have been marked as contributing to the Total Sobol index,
    # then normalize by variance
    VTi = (np.transpose(Fact_STi)) @ gpce_squared
    Tot_Sob_i = np.squeeze(VTi / Var)
    # repeat for the principal Sobol index
    Vi = (np.transpose(Fact_Si)) @ gpce_squared
    Sob_i = np.squeeze(Vi / Var)
    return Sob_i, Tot_Sob_i, Mean, Var
