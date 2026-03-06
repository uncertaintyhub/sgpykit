import numpy as np

from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.tools.lev2knots_functions import lev2knots_lin


def define_functions_for_rule(rule, input2):
    """
    Define the level-to-knots and index set functions for a given sparse grid rule.

    This function sets the functions `lev2nodes` and `idxset` to be used in
    `create_sparse_grid` to build the desired isotropic or anisotropic sparse grid.

    Parameters
    ----------
    rule : str
        The rule for constructing the sparse grid. Can be one of the following:
        'TP', 'TD', 'HC', 'SM'.
    input2 : int or array_like
        If scalar, the number of variables (N). If array_like, the rates vector
        for anisotropic sparse grids.

    Returns
    -------
    lev2nodes : function
        A function that maps a level to the number of knots.
    idxset : function
        A function that computes the index set for a given multi-index.

    Raises
    ------
    ValueError
        If the rule is unknown.
    """
    # [lev2nodes,idxset] = DEFINE_FUNCTIONS_FOR_RULE(rule,N)
    # sets the functions lev2nodes and idxset to use in create_sparse_grid.m to build the desired ISOTROPIC sparse grid.
    # N is the number of variables.
    # [lev2nodes,idxset] = DEFINE_FUNCTIONS_FOR_RULE(rule,rates)
    # sets the functions lev2nodes and idxset to use in create_sparse_grid.m to build the desired ANISOTROPIC sparse grid
    # with specified rates.
    # rule can be any of the following: 'TP', 'TD', 'HC', 'SM'
    # The outputs are anonymous function , lev2nodes =@(i) ... and idxset=@(i) ...
    # Determine rates
    rates = np.ones(input2) if np.isscalar(input2) else np.array(input2)

    # Define rule configurations
    rule_configs = {
        'TP': (lev2knots_lin, lambda i: np.max(rates[:len(i)] * i)),
        'TD': (lev2knots_lin, lambda i: np.sum(rates[:len(i)] * i)),
        'HC': (lev2knots_lin, lambda i: np.prod((i + 1) ** rates[:len(i)])),
        'SM': (lev2knots_doubling, lambda i: np.sum(rates[:len(i)] * i))
    }

    # Get configuration or raise error
    try:
        return rule_configs[rule]
    except KeyError:
        raise ValueError(f"Unknown rule: {rule}")
