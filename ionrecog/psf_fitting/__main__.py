"""Standalone alias: python -m ionrecog.psf_fitting --input ... --output ..."""
import sys

from ..cli import main

if __name__ == "__main__":
    main(["psf"] + sys.argv[1:])
