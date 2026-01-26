"""
Tests for the fetcher module.
"""

import pytest
from hypothesis import given, strategies as st
from nifty_analyzer.fetcher import fetch_stock_data, fetch_all_stocks

# TODO: Implement tests in tasks 4.3, 4.4, 4.5