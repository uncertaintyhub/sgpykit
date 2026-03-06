import logging
import numpy as np

from sgpykit.src.GK_lev_table import GK_lev_table
from sgpykit.util.checks import is_number_or_list_of_numbers_nonnegative

logger = logging.getLogger(__name__)


def lev2knots_GK(I):
    """
    Map sparse-grid levels to the corresponding number of knots using the Genz-Keister (GK) table.

    This function returns the number of knots corresponding to the given level(s) I,
    using the pre-tabulated GK_lev_table. For levels beyond the table's range,
    a warning is logged and -1 is stored as a sentinel value.

    Parameters
    ----------
    I : int or array_like
        Level(s) for which to compute the number of knots. Must be non-negative.

    Returns
    -------
    nb_knots : int or ndarray
        Number of knots corresponding to the input level(s). If I is a scalar, returns a scalar.
        If I is an array, returns an array of the same shape. For non-tabulated levels, -1 is stored.

    Notes
    -----
    The GK_lev_table is a pre-computed lookup table that maps levels to knot counts and quadrature orders.
    Levels greater than 5 are not tabulated and will result in a warning and a stored value of -1.
    """
    # nb_knots = lev2knots_GK(I)
    # returns the number of knots corresponding to the i-level i,
    # via the i2l map tabulated as in GK_lev_table
    assert is_number_or_list_of_numbers_nonnegative(I), f"Error: {I} ({type(I)})"
    isarray = hasattr(I, '__iter__')
    I = np.atleast_1d(I)
    mask = I > 5
    # are there non tabulated levels?
    if np.any(mask) == False:
        # nb knots is a vector after this instruction; I is understood coloumnwise as I(:)
        # Note that I want to access rows I+1, because minimum value for level is 0,
        # whose data are stored in row 1
        nb_knots = GK_lev_table(I, 2)
    else:
        logger.warning('GKNonTab: asking for non tabulated levels')
        I[mask] = 0
        # @NOTE: no nullable integer data type in numpy. pandas has one Int64 but is experimental and incomplete
        vect_nb_knots = GK_lev_table(I, 2)  # pd.array(GK_lev_table(I,2), dtype=pd.Int64Dtype())
        # @NOTE: np.inf does not work here (GK_lev_table are integers), would cast to float otherwise
        vect_nb_knots[mask] = -1
        nb_knots = vect_nb_knots

    return np.reshape(nb_knots, I.shape) if isarray else nb_knots[0]
