from sgpykit.util import matlab


def asreduced(pts_list, wgs_list=None):
    """
    Construct a reduced sparse grid structure from given points and weights.

    Parameters
    ----------
    pts_list : array_like
        List of knot points. The "knots" field of the returned
        sparse grid structure is set to this list.
    wgs_list : array_like, optional
        List of weights corresponding to the knot points. If provided, the
        "weights" field of the returned sparse grid structure is set to this list. The
        default is None.

    Returns
    -------
    S : struct
        A reduced sparse grid structure with the following fields:
        - knots : array_like
            The knot points.
        - weights : array_like
            The weights corresponding to the knot points.
        - size : int
            The number of knot points.
        - n : list
            Empty list.
        - m : list
            Empty list.

    Raises
    ------
    ValueError
        If the number of points and weights are different.

    Notes
    -----
    The knots in pts_list are assumed to be all different; the list is not
    checked for duplicates.
    """
    S = matlab.struct(knots=pts_list, n=[], m=[])

    if pts_list is not None and wgs_list is not None:
        if len(S.knots) == pts_list.shape[1]:
            S.weights = wgs_list
            S.size = len(S.weights)
        else:
            raise ValueError('the number of points and of weights are different')
    else:
        S.weights = []
        S.size = pts_list.shape[1]

    return S
