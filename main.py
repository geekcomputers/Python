#!/usr/bin/env python3
"""
Main entry point script for the Nifty 50 Stock Analyzer.
Can be executed directly as: python main.py
"""

import sys
from nifty_analyzer.cli import main

if __name__ == "__main__":
    sys.exit(main())