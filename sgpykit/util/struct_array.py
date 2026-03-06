import copy
import logging
import numpy as np

from sgpykit.util.cell import Cell
from sgpykit.util.misc import get_shape_of_cells
from sgpykit.util.struct import Struct

logger_ = logging.getLogger(__name__)


class StructArray:
    # TODO: this needs a refactoring
    #
    # https://de.mathworks.com/help/matlab/ref/struct.html
    # Matlab: non-scalar cells given via constructor makes it a StructArray, but not when given to a property later on.
    #         Use np.array instead of cell in the constructor.
    # oct2py backend StructArray implementation lacks even more features:
    # https://oct2py.readthedocs.io/en/latest/api.html
    def __init__(self, *args, **kwargs):
        """
        Initialize a StructArray.

        Parameters
        ----------
        *args : tuple
            Key-value pairs for struct fields.
        **kwargs : dict
            Key-value pairs for struct fields.

        Notes
        -----
        - If a single numeric argument is provided, it creates an array of empty Structs.
        - If key-value pairs are provided, it creates a StructArray with the given fields.
        - Supports both 1D and 2D shapes.
        """
        assert len(args) == 0 or len(kwargs) == 0
        self.struct = []
        self.shape = (0, 0)
        if len(args) == 1:
            assert np.issubdtype(type(args[0]), np.number), "A single argument can only be a number (do not pass multiple arguments as an array)"
            count = int(args[0])
            assert count > 0
            self.struct = [Struct() for _ in range(count)]
            self.shape = (1, count)
        elif len(kwargs) == 0 or len(args) == 0:
            # args: key, obj, key2, obj2, ...
            # kwargs: key=obj, key2=obj2, ...
            if len(kwargs) == 0:
                keys = args[::2]
                values = args[1::2]
                assert len(values) == len(keys)
            else:
                keys = kwargs.keys()
                values = kwargs.values()
            self.shape = get_shape_of_cells(values)
            assert self.shape
            if len(self.shape) == 1:
                self.shape = (1, self.shape[0])
            assert len(self.shape) == 2

            if self.shape[0] == 1 or self.shape[1] == 1:
                count = np.max(self.shape) # nr of cells
                self.struct = [Struct() for _ in range(count)]
                for i in range(count):
                    for key, value in zip(keys, values):
                        if isinstance(value, Cell):
                            val = value[i]
                            setattr(self.struct[i], key, copy.copy(val))  # set cell entry
                        else:
                            setattr(self.struct[i], key, copy.copy(value))  # duplicate objects to cells
            else:
                nrows, ncols = self.shape
                self.struct = [[Struct() for _ in range(ncols)] for _ in range(nrows)]
                for i in range(nrows):
                    for j in range(ncols):
                        for key, value in zip(keys, values):
                            if isinstance(value, Cell):
                                val = value[i,j]
                                setattr(self.struct[i][j], key, copy.copy(val))  # set cell entry
                            else:
                                setattr(self.struct[i][j], key, copy.copy(value))  # duplicate objects to cells

    # S.name
    def __getattr__(self, name: str):
        """
        Get attribute from all structs in the array.

        Parameters
        ----------
        name : str
            Name of the attribute to retrieve.

        Returns
        -------
        list or list of lists
            List of attribute values for 1D or 2D struct arrays.
        """
        if self.shape[0] == 1 or self.shape[1] == 1:
            rval = []
            count = np.max(self.shape) # nr of cells
            for i in range(count):
                if hasattr(self.struct[i], name):
                    rval.append(getattr(self.struct[i], name))
            return rval
        else: # 2D case
            nrows, ncols = self.shape
            retmat = []
            for i in range(nrows):
                retrow = []
                for j in range(ncols):
                    if hasattr(self.struct[i][j], name):
                        retrow.append(getattr(self.struct[i][j], name))
                retmat.append(retrow)
            return retmat

    # S[idx]
    def __getitem__(self, idx):
        """
        Get struct at index.

        Parameters
        ----------
        idx : int or str
            Index of the struct or name of the attribute.

        Returns
        -------
        Struct or attribute value
            Struct at index or attribute value.
        """
        if np.issubdtype(type(idx), np.number):
            return self.struct[idx]
        else:
            return getattr(self, idx)

    def __setitem__(self, idx, value):
        """
        Set struct at index.

        Parameters
        ----------
        idx : int
            Index of the struct.
        value : Struct
            Struct to set.
        """
        self.struct[idx] = value

    def __repr__(self):
        """
        String representation of the StructArray.

        Returns
        -------
        str
            String representation.
        """
        return f"StructArray({self.struct})"

    def __len__(self):
        """
        Length of the StructArray.

        Returns
        -------
        int
            Maximum dimension size.
        """
        # octave returned always the maximum for length() on a multi-index struct
        return np.max(self.shape)

    def isequal_to(self, obj2):
        """
        Check if two StructArrays are equal.

        Parameters
        ----------
        obj2 : StructArray
            StructArray to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        assert hasattr(obj2, 'struct')
        count = len(self)
        if len(obj2) != count:
            logger_.warning(f"StructArray length mismatch: {len(obj2)} != {count}")
            return False
        for i in range(count):
            if not self[i].isequal_to(obj2[i]):
                logger_.warning(f"StructArray item mismatch at index {i}")
                return False
        return True
