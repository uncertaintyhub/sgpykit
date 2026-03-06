import sys
from typing import Final

INT_NAN_WORKAROUND: Final = -sys.maxsize - 1  # soft limit, as python3+ int is unbounded
