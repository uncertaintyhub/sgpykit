import logging
import numpy as np

logger_ = logging.getLogger(__name__)
# TODO: refactor or replace redundant functions here


def contains_same_elements(obj1, obj2):
    """
    Check if two objects contain the same elements.

    Parameters
    ----------
    obj1 : array_like
        First object to compare.
    obj2 : array_like
        Second object to compare.

    Returns
    -------
    bool
        True if both objects contain the same elements, False otherwise.
    """
    return set(obj1) == set(obj2)


def is_numeric_scalar(item):
    """
    Check if an item is a numeric scalar.

    Parameters
    ----------
    item : object
        Item to check.

    Returns
    -------
    bool
        True if the item is a numeric scalar, False otherwise.
    """
    return isinstance(item, (int, float, complex, np.number))


def is_array_like(obj):
    """
    Check if an object is array-like.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.
    """
    return isinstance(obj, (list, np.ndarray))


def is_list_math_equal(list1, list2, tol=1e-10):
    """
    Recursively compares two nested lists or arrays to check if they are equal.

    Parameters
    ----------
    list1 : array_like
        First list or array to compare.
    list2 : array_like
        Second list or array to compare.
    tol : float, optional
        Tolerance for floating-point comparison. Default is 1e-10.

    Returns
    -------
    bool
        True if the lists or arrays are equal within the specified tolerance, False otherwise.

    Raises
    ------
    AssertionError
        If the lists or arrays are not equal.
    """

    def convert_to_array_like(obj):
        """
        Convert an object to an array-like object.

        Parameters
        ----------
        obj : object
            Object to convert.

        Returns
        -------
        np.ndarray
            Converted array-like object.

        Raises
        ------
        ValueError
            If the object type is unsupported.
        """
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, list):
            return np.array(obj, dtype=object)  # Use dtype=object to handle inhomogeneous arrays
        else:
            raise ValueError(f"Unsupported type: {type(obj)}")

    def is_element_equal(elem1, elem2, tol):
        """
        Check if two elements are equal within a specified tolerance.

        Parameters
        ----------
        elem1 : object
            First element to compare.
        elem2 : object
            Second element to compare.
        tol : float
            Tolerance for floating-point comparison.

        Returns
        -------
        bool
            True if the elements are equal within the specified tolerance, False otherwise.
        """
        if is_numeric_scalar(elem1) and is_numeric_scalar(elem2):
            if not np.isclose(elem1, elem2, atol=tol, rtol=0):
                logger_.warning(f"Scalars are not equal: {elem1} != {elem2}")
                return False
        elif np.isscalar(elem1) and np.isscalar(elem2):
            if not elem1 == elem2:
                logger_.warning(f"Scalars are not equal: {elem1} != {elem2} with types {type(elem1)}, {type(elem2)}")
                return False
        elif is_array_like(elem1) and is_array_like(elem2):
            return is_list_math_equal(elem1, elem2, tol)
        else:
            logger_.warning(f"Types are not the same: {type(elem1)} != {type(elem2)}")
            return False

    if not is_array_like(list1) or not is_array_like(list2):
        logger_.warning(f"Both inputs must be lists or NumPy arrays: {type(list1)} != {type(list2)}")
        return False

    if isinstance(list1, np.ndarray) and isinstance(list2, np.ndarray):
        if list1.shape != list2.shape:
            # try again as lists
            logger_.warning(f"Arrays have different shapes: {list1.shape} != {list2.shape}")
            return is_list_math_equal(list1.tolist(), list2.tolist())
        for sub_elem1, sub_elem2 in zip(list1, list2):
            if is_element_equal(sub_elem1, sub_elem2, tol) == False:
                return False
    elif isinstance(list1, list) and isinstance(list2, list):
        if len(list1) != len(list2):
            logger_.warning(f"Lists have different lengths: {len(list1)} != {len(list2)}")
            return False
        for sub_elem1, sub_elem2 in zip(list1, list2):
            if is_element_equal(sub_elem1, sub_elem2, tol) == False:
                return False
    else:
        # Convert both to NumPy arrays for comparison
        array1 = convert_to_array_like(list1)
        array2 = convert_to_array_like(list2)
        return is_list_math_equal(array1, array2, tol)
    return True


# TODO: replace np.number checks with more flexible isnumeric ones
def is_number_or_list_of_numbers_nonnegative(obj):
    """
    Check if an object is a non-negative number or a list of non-negative numbers.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if the object is a non-negative number or a list of non-negative numbers, False otherwise.
    """
    return hasattr(obj, '__iter__') == False and np.issubdtype(type(obj), np.number) and obj >= 0 or \
    hasattr(obj, '__iter__') and np.all([np.issubdtype(type(k), np.number) and k >= 0 for k in np.array(obj).flatten()])


def is_number_or_nparray_of_numbers_nonnegative(obj):
    """
    Check if an object is a non-negative number or a NumPy array of non-negative numbers.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if the object is a non-negative number or a NumPy array of non-negative numbers, False otherwise.
    """
    return np.issubdtype(type(obj), np.number) and obj >= 0 \
        or isinstance(obj, np.ndarray) and np.all([np.issubdtype(type(k), np.number) and k >= 0 for k in obj.flatten()])


def is_nparray_of_numbers(obj):
    """
    Check if an object is a NumPy array of numbers.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if the object is a NumPy array of numbers, False otherwise.
    """
    return isinstance(obj, np.ndarray) and np.all([np.issubdtype(type(k), np.number) for k in obj.flatten()])


def is_number_or_nparray_of_numbers(obj):
    """
    Check if an object is a number or a NumPy array of numbers.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if the object is a number or a NumPy array of numbers, False otherwise.
    """
    return np.issubdtype(type(obj), np.number) or is_nparray_of_numbers(obj)
