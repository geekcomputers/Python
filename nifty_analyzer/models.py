"""
Data models for the Nifty 50 Stock Analyzer.
Defines data structures for stock data and analysis results.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StockData:
    """Represents raw data fetched from Yahoo Finance."""
    symbol: str                          # Stock symbol (e.g., 'RELIANCE.NS')
    current_price: float                 # Most recent closing price
    historical_prices: List[float]       # Last 30 days of closing prices
    fetch_success: bool                  # Whether data fetch succeeded
    error_message: Optional[str] = None  # Error details if fetch failed

@dataclass
class AnalysisResult:
    """Represents analyzed stock with calculated metrics."""
    symbol: str                              # Stock symbol
    current_price: float                     # Current trading price
    moving_average_20d: Optional[float]      # 20-day moving average (None if insufficient data)
    percentage_difference: Optional[float]   # Percentage above/below MA
    is_highlighted: bool                     # True if >= 5% above MA
    error_message: Optional[str] = None      # Error details if analysis failed