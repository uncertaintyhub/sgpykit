from sgpykit.util import matlab


def is_tensor_grid(S):
    """
    Check if a given struct is a tensor grid or a tensor grid in a sparse grid struct.

    Parameters
    ----------
    S : struct
        Input struct to be checked.

    Returns
    -------
    int
        1 if S is a tensor grid, -1 if S is a tensor grid in a sparse grid struct, 0 otherwise.

    Notes
    -----
    A tensor grid is a struct with fields: 'knots', 'weights', 'size', 'knots_per_dim', 'm'.
    A tensor grid in a sparse grid struct is a struct with two more fields: 'coeff', 'idx'.
    """
    tensorgrid_fieldnames = {'knots', 'weights', 'size', 'knots_per_dim', 'm'}
    sparsegrid_fieldnames = tensorgrid_fieldnames.union({'coeff', 'idx'})
    ist = 0
    if matlab.isstruct(S) and len(S)==1:  # checks if S is Struct or StructArray
        s_fieldnames = set(matlab.fieldnames(S))
        if sparsegrid_fieldnames.issubset(s_fieldnames):
            ist = -1
        elif tensorgrid_fieldnames.issubset(s_fieldnames):
            ist = 1

    return ist
