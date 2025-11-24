#!/usr/bin/env python3
"""
Check and print installed versions of specified libraries.
"""

import sys
from importlib.metadata import version, PackageNotFoundError

packages = [
    "gym",
    "sortedcontainers",
    "tqdm",
    "gymnasium",
    "numpy",
    "torch",
    "torchvision",
    "opencv-python",
    "matplotlib",
    "tensorboard",
]

def main():
    for pkg in packages:
        try:
            ver = version(pkg)
            print(f"{pkg}: {ver}")
        except PackageNotFoundError:
            print(f"{pkg}: not installed")

if __name__ == "__main__":
    main()
