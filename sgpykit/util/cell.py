import numpy as np


class Cell(list):
    """
    A 1D or 2D container class that extends Python's built-in list.

    This class provides a way to store elements in a grid-like structure with
    support for both flat and tuple indexing. It is designed to mimic MATLAB's
    cell array functionality.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the cell. If an integer is provided, a square shape is assumed.
    dtype : type, optional
        The data type of the elements in the cell. Default is float.

    Attributes
    ----------
    shape : tuple
        The shape of the cell.
    dtype : type
        The data type of the elements in the cell.
    data : list
        The internal data structure storing the elements.
    """

    def __init__(self, shape, dtype=float):
        # Validate shape
        if isinstance(shape, int):
            shape = (shape, shape)
        if len(shape) > 2:
            raise ValueError("Only 1D and 2D cells are supported.")

        # Initialize the list with the specified shape
        self.shape = shape
        self.dtype = dtype
        if dtype is np.ndarray:
            self.data = [[list() for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            self.data = [[dtype() for _ in range(shape[1])] for _ in range(shape[0])]
        super().__init__(self.data)

    def __getitem__(self, index):
        """
        Get an item from the cell using either flat or tuple indexing.

        Parameters
        ----------
        index : int or tuple of ints
            The index of the item to retrieve. If an integer is provided, it is
            treated as a flat index and converted to a tuple.

        Returns
        -------
        object
            The item at the specified index.
        """
        if isinstance(index, int):
            # Flat indexing
            row, col = np.unravel_index(index, self.shape)
            return super().__getitem__(row)[col]
        else:
            row, col = index
            return super().__getitem__(row)[col]

    def __setitem__(self, index, value):
        """
        Set an item in the cell using either flat or tuple indexing.

        Parameters
        ----------
        index : int or tuple of ints
            The index of the item to set. If an integer is provided, it is
            treated as a flat index and converted to a tuple.
        value : object
            The value to set at the specified index.
        """
        if isinstance(index, int):
            # Flat indexing assignment
            row, col = np.unravel_index(index, self.shape)
            super().__getitem__(row)[col] = value
        else:
            row, col = index
            super().__getitem__(row)[col] = value

    def __len__(self):
        """
        Return the maximum dimension size of the cell.

        Returns
        -------
        int
            The maximum dimension size.
        """
        return np.max(self.shape)

    def __repr__(self):
        """
        Return a string representation of the cell.

        Returns
        -------
        str
            A string representation of the cell.
        """
        return f"Cell(shape={self.shape}, data={super().__repr__()})"

    def tolist(self):
        """
        Return the first row of the cell as a list.

        Returns
        -------
        list
            The first row of the cell.
        """
        return self.data[0]
