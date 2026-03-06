import logging

logger = logging.getLogger(__name__)


class WildcardImportWarning(list):
    """
    A custom list subclass that logs a warning when iterated over.

    This class is used to prevent wildcard imports from subpackages that contain
    subpackages. When iterated, it logs a warning and returns an empty iterator.

    Attributes
    ----------

    Methods
    -------
    __iter__() :
        Logs a warning about unsupported wildcard imports and returns an empty iterator.
    """
    def __iter__(self):
        logger.warning("Wildcard import is only supported for subpackages that do not contain subpackages."
                      " Nothing was imported.", UserWarning)
        return iter([])

__all__ = WildcardImportWarning()
