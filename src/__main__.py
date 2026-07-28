"""Allow `python -m src` as an alias for `python -m src.pipeline`."""

import sys

from .pipeline import main

if __name__ == "__main__":
    sys.exit(main())
