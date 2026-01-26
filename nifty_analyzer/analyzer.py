"""
Analysis engine for technical calculations.
Performs moving average calculations and identifies stocks meeting threshold criteria.
"""

from typing import List, Optional
from .models import StockData, AnalysisResult

def calculate_moving_average(prices: List[float], period: int = 20) -> Optional[float]:
    """
    Calculates simple moving average.
    Args:
        prices: List of closing prices (most recent last)
        period: Number of periods for moving average
    Returns: Moving average value or None if insufficient data
    """
    # TODO: Implement moving average calculation in task 5
    return None

def calculate_percentage_difference(current: float, average: float) -> float:
    """
    Calculates percentage difference between current price and moving average.
    Args:
        current: Current stock price
        average: Moving average value
    Returns: Percentage difference using formula: ((current - average) / average) * 100
    """
    # TODO: Implement percentage calculation in task 5
    return 0.0

def analyze_stock(stock_data: StockData) -> AnalysisResult:
    """
    Performs complete analysis on stock data.
    Args:
        stock_data: StockData object from fetcher
    Returns: AnalysisResult with all calculated metrics
    """
    # TODO: Implement stock analysis in task 5
    return AnalysisResult(
        symbol=stock_data.symbol,
        current_price=stock_data.current_price,
        moving_average_20d=None,
        percentage_difference=None,
        is_highlighted=False,
        error_message="Not implemented yet"
    )