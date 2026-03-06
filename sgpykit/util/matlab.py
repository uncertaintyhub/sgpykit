from collections.abc import Sequence

import numpy as np
from scipy.io import savemat

from sgpykit.util.cell import Cell
from sgpykit.util.checks import is_numeric_scalar
from sgpykit.util.misc import get_shape_of_cells
from sgpykit.util.struct import Struct
from sgpykit.util.struct_array import StructArray


def struct(*args, **kwargs):
    """
    Create a Struct or StructArray from given arguments.

    This function mimics MATLAB's struct behavior. If keyword arguments are provided,
    it creates a Struct with the given fields. If positional arguments are provided,
    it creates a Struct with fields from the even-indexed arguments and values from
    the odd-indexed arguments. If cells are detected, it creates a StructArray.

    Parameters
    ----------
    *args : tuple
        Positional arguments for field-value pairs.
    **kwargs : dict
        Keyword arguments for field-value pairs.

    Returns
    -------
    Struct or StructArray
        The created struct or struct array.
    """
    if len(kwargs) == 0 and len(args) > 1 or len(kwargs) > 0:
        # like matlab: check if there are cells given, otherwise return a single Struct
        if len(kwargs) > 0:
            cellshape = get_shape_of_cells(kwargs.values())
        else:
            cellshape = get_shape_of_cells(args[1::2])
        if cellshape is None:
            if len(kwargs) > 0:
                return Struct(**kwargs)
            else:
                return Struct(**dict(zip(args[::2], args[1::2])))
    return StructArray(*args, **kwargs)


def fieldnames(s):
    """
    Get the field names of a Struct or StructArray.

    Parameters
    ----------
    s : Struct or StructArray
        The struct or struct array.

    Returns
    -------
    list
        The list of field names.
    """
    if isinstance(s, Struct):
        return list(vars(s).keys())
    elif hasattr(s, 'shape'):
        if s.shape[0] > 1:  # 2D case
            return list(vars(s.struct[0][0]).keys())
        else:
            return list(vars(s.struct[0]).keys())
    else:
        return list(vars(s.struct).keys())


def cell(shape):
    """
    Returns an empty cell (numpy) array with the given shape (empty nD-matrix).

    2D cell objects can be flat indexed, e.g. c[0,2] or c[2].
    Higher dimensions than 2D are not supported.
    If shape is a single integer N, create a NxN cell

    Parameters
    ----------
    shape : int or tuple
        The shape of the cell array. If an integer, creates a square cell array.

    Returns
    -------
    Cell
        The empty cell array.
    """
    if isinstance(shape, int):
        return Cell((shape, shape))
    return Cell(shape)


def ce(*args):
    """
    Replacement for the matlab {} operator to create cells.
    The cell elements must be of same data type.

    Parameters
    ----------
    *args : tuple
        Elements to be placed in the cell array.

    Returns
    -------
    Cell
        The created cell array.
    """
    result_cell = Cell((1, len(args)), dtype=type(args[0]))
    for i, arg in enumerate(args):
        result_cell[i] = arg
    return result_cell


def iscell(var):
    """
    Check if the given variable is a Cell.

    Parameters
    ----------
    var : any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a Cell, False otherwise.
    """
    return isinstance(var, Cell)


def isstruct(var):
    """
    Check if the given variable is an instance of Struct or StructArray.

    Parameters
    ----------
    var : any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an instance of Struct or StructArray, False otherwise.
    """
    return isinstance(var, Struct) or isinstance(var, StructArray)


def isfield(var, sel):
    """
    Check if the given field exists in the struct.

    Parameters
    ----------
    var : str
        The field name to check.
    sel : Struct or StructArray
        The struct or struct array.

    Returns
    -------
    bool
        True if the field exists, False otherwise.
    """
    return var in fieldnames(sel)


def unique(var, by=None):
    """
    Incomplete matlabs unique() function.

    https://de.mathworks.com/help/matlab/ref/double.unique.html
    - 'first' and 'sorted' are defaults
    https://numpy.org/doc/stable/reference/generated/numpy.unique.html#numpy-unique

    Parameters
    ----------
    var : array_like
        Input array.
    by : str, optional
        The axis to return indices for repetitions. Default is None.

    Returns
    -------
    tuple
        The unique elements and their indices.
    """
    if by == 'first':  # which indices to return if repetitions
        by = 0
    elif by == 'row':
        by = 0

    return np.unique(var, axis=by, return_index=True)


def find(X, n=None, direction='first'):
    """
    Find non-zero elements in an array.

    Parameters
    ----------
    X : array_like
        Input array.
    n : int, optional
        Maximum number of elements to return. Default is None.
    direction : str, optional
        Direction to search ('first' or 'last'). Default is 'first'.

    Returns
    -------
    tuple
        Indices and values of non-zero elements.
    """
    # https://www.mathworks.com/help/matlab/ref/find.html?searchHighlight=find
    if n is None:
        return np.argwhere(X)
    ## Find non-zero elements
    idx = np.nonzero(X)[0]
    values = X[idx]

    ## Sort based on direction
    if direction == 'last':
        values = values[::-1]

    ## Limit to n elements if specified
    if n is not None:
        n = min(n, len(values))
        idx = idx[:n]
        values = values[:n]

    return idx, values


def setdiff(A, B, kind='rows'):
    """
    Set difference of two arrays.

    Parameters
    ----------
    A : array_like
        First input array.
    B : array_like
        Second input array.
    kind : str, optional
        Type of set difference ('rows' or others). Default is 'rows'.

    Returns
    -------
    ndarray
        The set difference.
    """
    #return np.setdiff1d(var1, var2)
    if 'rows' in kind:
        # Ensure A and B are numpy arrays
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        # Check if the number of columns in A and B are the same
        if A.shape[1] != B.shape[1]:
            raise ValueError("A and B must have the same number of columns.")

        # Use a list comprehension to filter rows in A not in B
        diff = np.array([row for row in A if not any(np.array_equal(row, b_row) for b_row in B)])
        return diff
    else:
        raise NotImplementedError("Not implemented yet.")


def sortrows(A):
    """
    Sort rows of a matrix in ascending order.

    Parameters
    ----------
    A : array_like
        Input array.

    Returns
    -------
    tuple
        Sorted array and sorted indices.
    """
    # matlab doc on sortrows(A):
    # B = sortrows(A) sorts the rows of a matrix in ascending order based on the elements in the first column.
    # When the first column contains repeated elements, sortrows sorts according to the values in the next column and
    # repeats this behavior for succeeding equal values.
    if len(A) < 2:
        return A, [0]
    if np.atleast_2d(A).shape[0] == 1:
        return A, [0]
    # Reverse the order of columns for lexsort to sort by the first column first
    sorted_indices = np.lexsort(A.T[::-1])
    # Use these indices to sort the entire array
    sorted_array = A[sorted_indices]
    return sorted_array, sorted_indices


def sort(arr, axis=None, direction='ascend'):
    """
    Sort an array in ascending or descending order.

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, optional
        Axis along which to sort. Default is None.
    direction : str, optional
        Direction to sort ('ascend' or 'descend'). Default is 'ascend'.

    Returns
    -------
    tuple
        Sorted array and sorted indices.
    """
    if direction == 'ascend':
        indices = np.argsort(arr, axis=axis)
    elif direction == 'descend':
        indices = np.argsort(-arr, axis=axis)
    else:
        raise ValueError("direction must be 'ascend' or 'descend'")

    sorted_arr = np.take_along_axis(arr, indices, axis=axis)
    return sorted_arr, indices


def reshape(var, r, c=None): # TODO: to be removed
    """
    Reshape an array.

    Parameters
    ----------
    var : array_like
        Input array.
    r : int or array_like
        New shape or number of rows.
    c : int, optional
        Number of columns. Default is None.

    Returns
    -------
    ndarray
        Reshaped array.
    """
    # Convert input to numpy array if it's not already
    arr = np.array(var)

    # Flatten the input array
    # @TODO: can we just use array.reshape() ?
    flat_arr = arr.flatten()
    if c is None:
        if isinstance(r, np.ndarray) or isinstance(r, list):
            return flat_arr.reshape(r)
        else:
            raise ValueError(f"Cannot reshape array of size {flat_arr.size} into shape ({r}) (is {r} a list or nparray?)")
    # Check if the total number of elements matches r * c
    if flat_arr.size != r * c:
        raise ValueError(f"Cannot reshape array of size {flat_arr.size} into shape ({r}, {c})")

    # Reshape the flattened array
    return flat_arr.reshape((r, c))


def logical(var):
    """
    Convert array to boolean (0=false, else=true).

    Parameters
    ----------
    var : array_like
        Input array.

    Returns
    -------
    ndarray
        Boolean array.
    """
    # converts array to boolean (0=false, else=true)
    return np.array(var, dtype=bool)


def intersect(x, y):
    """
    Find the intersection of two arrays.

    Parameters
    ----------
    x : array_like
        First input array.
    y : array_like
        Second input array.

    Returns
    -------
    tuple
        Common elements and their indices in the original arrays.
    """
    # Ensure inputs are NumPy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Find unique elements in both arrays
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    # Find common elements
    common = np.intersect1d(x_unique, y_unique)

    # Find indices in original arrays
    x_indices = np.searchsorted(x_unique, common)
    y_indices = np.searchsorted(y_unique, common)

    return common, x_indices, y_indices


def issorted(matrix, typ='rows'):
    """
    Check if the matrix is sorted.

    Parameters
    ----------
    matrix : array_like
        Input matrix.
    typ : str, optional
        Type of sorting to check. Default is 'rows'.

    Returns
    -------
    bool
        True if the matrix is sorted, False otherwise.
    """
    # Check if the matrix is empty
    if matrix.size == 0:
        return True
    if typ != 'rows':
        raise Exception('Not implemented.')

    # Lexicographical sort check using numpy's lexsort
    # lexsort returns index list, and they should match index list 1:n when already sorted
    return np.all(np.lexsort(matrix.T[::-1]) == np.arange(len(matrix)))


# TODO: remove?
def fliplr(arr):
    """
    Flip the elements of the array from left to right.

    Parameters
    ----------
    arr : array_like
        Input array to be flipped.

    Returns
    -------
    ndarray
        Flipped array.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.ndim == 1:
        # For 1D arrays, simply reverse the array
        return arr[::-1]
    elif arr.ndim == 2:
        # For 2D arrays, flip each row from left to right
        return arr[:, ::-1]
    else:
        raise ValueError("Input array must be 1D or 2D.")


def ifft(arr):
    """
    Compute the inverse FFT of a 1D or 2D array, matching MATLAB's ifft() behavior.

    Parameters
    ----------
    arr : array_like
        A 1D or 2D numpy array.

    Returns
    -------
    ndarray
        The inverse FFT of the input array.

    Raises
    ------
    ValueError
        If the input array is not 1D or 2D.
    """
    arr = np.atleast_1d(np.squeeze(arr))
    if arr.ndim == 1:
        # For 1D array, compute the inverse FFT
        return np.fft.ifft(arr)
    elif arr.ndim == 2:
        # For 2D array, compute the inverse FFT of each column like MATLAB
        return np.fft.ifft(arr, axis=0)
    else:
        raise ValueError("Input array must be 1D or 2D.")


def eig(var):
    """
    Compute the eigenvalues and eigenvectors of a matrix.

    Parameters
    ----------
    var : array_like
        Input matrix.

    Returns
    -------
    tuple
        Eigenvectors and eigenvalues.
    """
    # X are eigenvalues (already a vector of the diagonals)
    # matlab would return X as matrix
    x, W = np.linalg.eig(var)
    return W, x  # swap to get matlab behavior


def isnumeric(obj):
    """
    Check if the object is numeric.

    Parameters
    ----------
    obj : any
        The object to check.

    Returns
    -------
    bool
        True if the object is numeric, False otherwise.
    """
    # Check if the object is a NumPy array
    if isinstance(obj, np.ndarray):
        # Check if the dtype of the array is a numeric type
        return np.issubdtype(obj.dtype, np.number)

    # Check if the object is a single number (int, float, etc.)
    if is_numeric_scalar(obj):
        return True

    # Check if the object is a sequence (list, tuple, etc.) and all elements are numbers
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return all(is_numeric_scalar(item) for item in obj)

    return False


def save(filename, *args):
    """
    Save NumPy arrays to a MATLAB file.

    Parameters
    ----------
    filename : str
        The name of the file to save.
    *args : tuple
        Variable length argument list of strings and NumPy arrays.
        The first argument is the filename, followed by pairs of variable names and arrays.
    """
    # Create a dictionary to hold the data
    data_dict = {}

    # Iterate over the arguments in pairs
    for i in range(0, len(args), 2):
        var_name = args[i]
        var_array = args[i + 1]
        data_dict[var_name] = var_array

    # Save the dictionary to a MATLAB file
    savemat(filename, data_dict)
