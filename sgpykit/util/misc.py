import itertools
import time
from contextlib import contextmanager
import numpy as np

from sgpykit.util.cell import Cell
from sgpykit.util.checks import is_numeric_scalar
from sgpykit.util.definitions import INT_NAN_WORKAROUND


def get_shape_of_cells(objects):
    """
    Get the shape of the first Cell object in the list of objects.

    Parameters
    ----------
    objects : list
        List of objects to check for Cell type and shape.

    Returns
    -------
    numpy.ndarray or None
        Shape of the first Cell object found, or None if no Cell object is found.
    """
    cellshape = None
    for obj in objects:
        cellshape_tmp = None
        if isinstance(obj, Cell):
            cellshape_tmp = obj.shape
        if cellshape is None and cellshape_tmp:
            cellshape = cellshape_tmp
        elif cellshape and cellshape_tmp and np.all(cellshape_tmp != cellshape):
            raise Exception('Cells must have same shape.')
    return cellshape


def generate_multi_indices(shape, min_idx=0):
    """
    Generate all multi-indices for a given shape.

    Parameters
    ----------
    shape : tuple
        Shape of the multi-indices to generate.
    min_idx : int, optional
        Minimum index value (default is 0).

    Returns
    -------
    numpy.ndarray
        Array of multi-indices.
    """
    # Create a list of arrays, each containing the indices for one dimension
    ranges = [np.arange(min_idx, dim + 1) for dim in shape]
    # Use meshgrid to create coordinate matrices, then flatten and stack them
    grids = np.meshgrid(*ranges, indexing='ij')
    indices = np.stack(grids, axis=-1).reshape(-1, len(shape))
    return indices


def matlab_to_python_index(var):
    """
    Convert MATLAB-style index to Python-style index.

    Parameters
    ----------
    var : int or numpy.ndarray
        MATLAB-style index or array of indices.

    Returns
    -------
    int or numpy.ndarray
        Python-style index or array of indices.
    """
    if is_numeric_scalar(var):
        return var-1
    else:
        return np.asarray(var)-1


def python_to_matlab_index(var):
    """
    Convert Python-style index to MATLAB-style index.

    Parameters
    ----------
    var : int or array-like
        Python-style index or array of indices.

    Returns
    -------
    int or numpy.ndarray
        MATLAB-style index or array of indices.
    """
    if is_numeric_scalar(var):
        return var+1
    else:
        return np.asarray(var)+1


def append(some_list, item):
    """
    Append an item to a list and return as a numpy array.

    Parameters
    ----------
    some_list : list
        List to append to.
    item : any
        Item to append.

    Returns
    -------
    numpy.ndarray
        Array with the item appended.
    """
    return np.array(some_list + [item])


def to_array(item, n):
    """
    Create an array of length n with the given item.

    Parameters
    ----------
    item : any
        Item to repeat.
    n : int
        Length of the array.

    Returns
    -------
    list
        Array of length n with the given item.
    """
    arr = [item]*n #Cell((1, N))
    return arr


def reshape_nested_lists_to_nrows(obj, nrows=2):
    """
    Reshape nested lists to a specified number of rows.

    Parameters
    ----------
    obj : list
        Nested list to reshape.
    nrows : int, optional
        Number of rows (default is 2).

    Returns
    -------
    list
        Reshaped list.
    """
    # reshapes a Kx Rows x Cols_{Row} to a Rows x AllCols list
    return [list(itertools.chain(*[sublist[i] for sublist in obj])) for i in range(nrows)]


def apply_lev2knots(i, lev2knots, N=None):
    """
    Apply level-to-knots mapping to an index array.

    Parameters
    ----------
    i : numpy.ndarray
        Array of 0-based indices.
    lev2knots : list or Cell or callable
        Level-to-knots mapping function or list of functions.
    N : int, optional
        Number of dimensions (default is None).

    Returns
    -------
    numpy.ndarray
        Mapped indices.
    """
    # m = apply_lev2knots(i,lev2knots,N)

    # return a vector m s.t. m(n) = m(i[n]). m has to be single (or double) precision or following functions won't work
    # (more specifically, prod is not defined for intXY classes)

    # N could be deduced by N but it's better passed as an input, to speed computation
    if N is None:
        N = i.shape[0]
    # init m to zero vector
    m = np.empty(i.shape[0], dtype=np.int64) # 0*i
    # next, iterate on each direction 1,...,N.

    if type(lev2knots) is list or type(lev2knots) is Cell:
        for n in range(N):
            m[n] = int(lev2knots[n](i[n]+1))
    else:
        for n in range(N):
            m[n] = int(lev2knots(i[n]+1))

    return m


def safe_vstack(*arrays):
    """
    Safely stack arrays vertically, filtering out empty arrays.

    Parameters
    ----------
    *arrays : numpy.ndarray
        Arrays to stack.

    Returns
    -------
    numpy.ndarray
        Stacked array.
    """
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]

    if not non_empty_arrays:
        # If all arrays are empty, return an empty array with shape (0,)
        return np.empty((0,))

    return np.vstack(non_empty_arrays)


def unique_rows_stable(A, B):
    """
    Find unique rows in the combined array of A and B, maintaining order.

    Parameters
    ----------
    A : numpy.ndarray
        First array.
    B : numpy.ndarray
        Second array.

    Returns
    -------
    numpy.ndarray
        Array of unique rows.
    """
    # Concatenate A and B along the first axis (vertically)
    combined = safe_vstack(A, B)

    # Use np.unique with return_index to get unique rows and their first occurrence indices
    _, idx = np.unique(combined, axis=0, return_index=True)

    # Sort the indices to maintain the original order
    idx_sorted = np.sort(idx)

    # Return the unique rows in the original order
    return combined[idx_sorted]


def is_int_nan(values):
    """
    Check if values are the integer NaN workaround.

    Parameters
    ----------
    values : numpy.ndarray
        Values to check.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating which values are the integer NaN workaround.
    """
    # TODO: workaround (internal NaN is defined as a large negative integer)
    return values==INT_NAN_WORKAROUND


def merge_all_args_to_kwargs(args, kwargs, to_lowercase=False):
    """
    Merge positional and keyword arguments into a single dictionary.

    Parameters
    ----------
    args : list
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    to_lowercase : bool, optional
        Convert keys to lowercase (default is False).

    Returns
    -------
    tuple
        Tuple of merged keyword arguments and format string.
    """
    fmt = None
    all_args = [*args]
    for key, value in kwargs.items():
        all_args.append(key)
        all_args.append(value)
    if np.mod(len(all_args), 2) == 1:  # odd items, extract fmt, otherwise we have matching key-value pairs
        fmt = all_args[0]
        all_args = all_args[1:]
    kwargs_all = dict(zip(all_args[::2], all_args[1::2]))

    if to_lowercase:
        # Convert all keys to lowercase strings
        kwargs_all = {str(key).lower(): value for key, value in kwargs_all.items()}

    return kwargs_all, fmt


def print_plain_int(x):
    """
    Print integer values in a plain format.

    Parameters
    ----------
    x : int or list or tuple
        Integer or list/tuple of integers to print.
    """
    if isinstance(x, (list, tuple)):
        print([int(i) for i in x])
    else:
        print(x)


@contextmanager
def time_block(description = ""):
    """Measure and print the execution time of a code block.

    Args:
        description (str): Optional description of the timed block.
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if description:
        print(f"[{description}] Elapsed time: {elapsed_time:.6f} seconds")
    else:
        print(f"Elapsed time: {elapsed_time:.6f} seconds")