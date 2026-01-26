# Nifty 50 Stock Analyzer

A Python CLI tool that analyzes Nifty 50 stocks in real-time, calculates 20-day moving averages, and identifies stocks trading significantly above their moving average.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analyzer:
```bash
python main.py
# or
python -m nifty_analyzer
```

## Features

- Fetches real-time data for all Nifty 50 stocks using yfinance
- Calculates 20-day moving averages
- Identifies stocks trading 5% or more above their moving average
- Exports results to timestamped CSV files
- Robust error handling with retry logic

## Project Structure

```
nifty_analyzer/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point for python -m nifty_analyzer
├── cli.py              # CLI interface and user interaction
├── symbols.py          # Nifty 50 stock symbol provider
├── models.py           # Data models (StockData, AnalysisResult)
├── fetcher.py          # Yahoo Finance data fetching with retry logic
├── analyzer.py         # Technical analysis calculations
└── exporter.py         # CSV export functionality

tests/
├── test_symbols.py     # Unit tests for symbol provider
├── test_models.py      # Property tests for data models
├── test_fetcher.py     # Tests for data fetching
├── test_analyzer.py    # Tests for analysis engine
├── test_exporter.py    # Tests for CSV export
├── test_cli.py         # Tests for CLI interface
└── test_integration.py # End-to-end integration tests
```

## Development Status

This project is currently under development. Individual modules will be implemented according to the task list in the specification document.