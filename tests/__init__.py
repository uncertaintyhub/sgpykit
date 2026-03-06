import logging

logger = logging.getLogger(__name__)


class WildcardImportWarning(list):
    def __iter__(self):
        logger.warning("Wildcard import is not supported for test."
                      " Nothing was imported.", UserWarning)
        return iter([])

__all__ = WildcardImportWarning()