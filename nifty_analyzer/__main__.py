"""
Main entry point for the nifty-stock-analyzer CLI tool.
Allows the package to be executed as: python -m nifty_analyzer
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())