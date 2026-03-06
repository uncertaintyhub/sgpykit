def lev2knots_lin(I):
    """
    Relation between level and number of points for linear level-to-knots mapping.

    This function maps a given level `I` to the number of points `m` by simply
    returning the same integer (i.e., `m = I`). It is intended for use in
    constructing sparse grids where a linear level-to-knots mapping is needed.

    Parameters
    ----------
    I : int or array-like
        Level(s) for which to compute the number of knots.

    Returns
    -------
    m : int or array-like
        Number of points to be used in each direction.

    Notes
    -----
    The relationship between level and number of points is given by:
        m = I
    """
    m = I
    return m
