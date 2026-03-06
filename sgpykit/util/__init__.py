import logging

logger_ = logging.getLogger(__name__)


class WildcardImportWarning(list):
    """
    Custom list subclass that logs a warning when wildcard import is attempted.

    This class is used to prevent wildcard imports from the util module by
    logging a warning and returning an empty iterator.

    Attributes
    ----------

    Methods
    -------
    __iter__() :
        Logs a warning about unsupported wildcard import and returns an empty iterator.
    """
    def __iter__(self):
        logger_.warning("Wildcard import is currently not supported for util."
                      " Nothing was imported.", UserWarning)
        return iter([])

__all__ = WildcardImportWarning()
