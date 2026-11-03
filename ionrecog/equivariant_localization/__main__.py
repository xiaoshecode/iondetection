"""Allow python -m ionrecog.equivariant_localization train/infer."""
import sys
from ..cli import main

if __name__ == "__main__":
    main(["equivariant"] + sys.argv[1:])
