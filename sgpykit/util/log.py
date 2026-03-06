import logging

"""
Example usage for a verbose output:
import logging
from sgpykit.util.log import logger

logging.basicConfig(
    # see https://docs.python.org/3/library/logging.html#logrecord-attributes
    format="%(levelname)s %(name)s:%(lineno)d: %(message)s",
)
# switching log level for sgpykit. Also see https://docs.python.org/3/library/logging.html#logging-levels
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
"""
logger = logging.getLogger("sgpykit")
logger.addHandler(logging.NullHandler())  # avoid "No handlers could be found" warnings
