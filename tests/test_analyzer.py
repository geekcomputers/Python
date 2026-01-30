"""
Tests for the analyzer module.
"""

import pytest
from hypothesis import given, strategies as st
from nifty_analyzer.analyzer import calculate_moving_average, calculate_percentage_difference, analyze_stock

# TODO: Implement tests in tasks 5.3, 5.4, 5.5, 5.6