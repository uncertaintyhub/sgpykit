import logging
import numpy as np

from sgpykit.util.checks import is_list_math_equal

logger_ = logging.getLogger(__name__)


class Struct:
    """
    A simple struct-like class that allows attribute access and dictionary-like item access.

    This class is designed to mimic MATLAB struct behavior, allowing flexible field access
    and comparison operations.

    Attributes
    ----------
    **kwargs : dict
        Arbitrary keyword arguments that become attributes of the struct.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Struct with the given keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to be stored as attributes.
        """
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        """
        Set an item using dictionary-like syntax.

        Parameters
        ----------
        key : str
            The name of the attribute to set.
        value : object
            The value to assign to the attribute.
        """
        self.__dict__[key] = value

    def __getitem__(self, key):
        """
        Get an item using dictionary-like syntax.

        Parameters
        ----------
        key : str
            The name of the attribute to retrieve.

        Returns
        -------
        object
            The value of the attribute.
        """
        return self.__dict__[key]

    def __len__(self):
        """
        Return the length of the struct (always 1).

        Returns
        -------
        int
            The length of the struct.
        """
        return 1

    def __repr__(self):
        """
        Return a string representation of the struct.

        Returns
        -------
        str
            A formatted string showing the struct's fields and values.
        """
        content = ', '.join('\n\n'+f'{k}={v}' for k, v in self.__dict__.items())
        return f"Struct:{content}"+"\n======\n"

    def fieldnames(self):
        """
        Return a list of the struct's field names.

        Returns
        -------
        list
            A list of the struct's field names.
        """
        return list(vars(self).keys())

    def isequal_to(self, obj2):
        """
        Check if this struct is equal to another struct-like object.

        Parameters
        ----------
        obj2 : Struct or StructArray
            The structure object to compare with.

        Returns
        -------
        bool
            True if the structs are equal, False otherwise.
        """
        assert hasattr(obj2, 'fieldnames')
        fms1 = self.fieldnames()
        fms2 = obj2.fieldnames()
        if fms1 != fms2:
            logger_.warning(f"fieldnames mismatch: {fms1} != {fms2}")
            return False
        for field in fms1:
            if np.isscalar(self[field]):
                if self[field] != obj2[field]:
                    logger_.warning(f"... at field={field}")
                    return False
            elif not is_list_math_equal(self[field], obj2[field]):
                logger_.warning(f"... at field={field}")
                return False
        return True
