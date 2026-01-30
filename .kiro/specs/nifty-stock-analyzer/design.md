# Design Document: Nifty 50 Stock Analyzer

## Overview

The Nifty 50 Stock Analyzer is a Python CLI application that provides real-time technical analysis of India's top 50 stocks. The tool leverages the yfinance library to fetch market data, performs moving average calculations, and identifies momentum opportunities by highlighting stocks trading significantly above their 20-day moving average. Results are exported to CSV format for further analysis.

The application follows a simple pipeline architecture: data retrieval → calculation → filtering → export, with robust error handling at each stage.

## Architecture

The application uses a modular, pipeline-based architecture:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   CLI       │────▶│   Data       │────▶│  Analysis   │────▶│   Export     │
│  Interface  │     │  Fetcher     │     │  Engine     │     │   Handler    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           │                     │                    │
                           ▼                     ▼                    ▼
                    ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
                    │  yfinance    │     │  Calculator │     │  CSV Writer  │
                    │   API        │     │   Module    │     │              │
                    └──────────────┘     └─────────────┘     └──────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Fail-Safe Operation**: Errors in processing individual stocks don't halt the entire analysis
3. **User Feedback**: Progress indicators and clear messaging throughout execution
4. **Data Integrity**: Validation at each pipeline stage ensures reliable results

## Components and Interfaces

### 1. CLI Interface Module (`cli.py`)

**Responsibility**: Entry point for the application, handles user interaction and orchestrates the pipeline.

**Key Functions**:
- `main()`: Entry point that coordinates the entire workflow
- `display_progress(current, total, stock_symbol)`: Shows progress during data fetching
- `display_summary(results)`: Prints analysis summary to console

**Interface**:
```python
def main() -> int:
    """
    Main entry point for the CLI tool.
    Returns: Exit code (0 for success, 1 for failure)
    """
```

### 2. Stock Symbol Provider (`symbols.py`)

**Responsibility**: Provides the list of Nifty 50 stock symbols with proper Yahoo Finance formatting.

**Key Functions**:
- `get_nifty50_symbols()`: Returns list of all Nifty 50 stock symbols

**Interface**:
```python
def get_nifty50_symbols() -> List[str]:
    """
    Returns the list of Nifty 50 stock symbols in Yahoo Finance format.
    Returns: List of stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS', ...])
    """
```

**Note**: Nifty 50 symbols require the `.NS` suffix for Yahoo Finance (National Stock Exchange of India).

### 3. Data Fetcher Module (`fetcher.py`)

**Responsibility**: Retrieves stock data from Yahoo Finance with retry logic and error handling.

**Key Functions**:
- `fetch_stock_data(symbol, days=30)`: Fetches historical data for a single stock
- `fetch_all_stocks(symbols)`: Fetches data for all provided symbols with progress tracking

**Interface**:
```python
@dataclass
class StockData:
    symbol: str
    current_price: float
    historical_prices: List[float]  # Last 30 days of closing prices
    fetch_success: bool
    error_message: Optional[str] = None

def fetch_stock_data(symbol: str, days: int = 30, retries: int = 3) -> StockData:
    """
    Fetches stock data with retry logic.
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        days: Number of days of historical data to fetch
        retries: Number of retry attempts on failure
    Returns: StockData object with fetched information
    """
```

### 4. Analysis Engine (`analyzer.py`)

**Responsibility**: Performs moving average calculations and identifies stocks meeting the threshold criteria.

**Key Functions**:
- `calculate_moving_average(prices, period=20)`: Computes moving average
- `calculate_percentage_difference(current, average)`: Computes percentage difference
- `analyze_stock(stock_data)`: Performs complete analysis on a single stock

**Interface**:
```python
@dataclass
class AnalysisResult:
    symbol: str
    current_price: float
    moving_average_20d: Optional[float]
    percentage_difference: Optional[float]
    is_highlighted: bool  # True if >= 5% above MA
    error_message: Optional[str] = None

def calculate_moving_average(prices: List[float], period: int = 20) -> Optional[float]:
    """
    Calculates simple moving average.
    Args:
        prices: List of closing prices (most recent last)
        period: Number of periods for moving average
    Returns: Moving average value or None if insufficient data
    """

def analyze_stock(stock_data: StockData) -> AnalysisResult:
    """
    Performs complete analysis on stock data.
    Args:
        stock_data: StockData object from fetcher
    Returns: AnalysisResult with all calculated metrics
    """
```

### 5. Export Handler (`exporter.py`)

**Responsibility**: Writes analysis results to CSV file with proper formatting.

**Key Functions**:
- `export_to_csv(results, filename)`: Writes results to CSV file
- `generate_filename()`: Creates timestamped filename

**Interface**:
```python
def export_to_csv(results: List[AnalysisResult], filename: Optional[str] = None) -> str:
    """
    Exports analysis results to CSV file.
    Args:
        results: List of AnalysisResult objects
        filename: Optional custom filename (generates timestamped name if None)
    Returns: Path to created CSV file
    """
```

**CSV Format**:
```
Symbol,Current Price,20-Day MA,Percentage Difference,Highlighted
RELIANCE.NS,2450.50,2350.25,4.26,No
TCS.NS,3500.00,3300.00,6.06,Yes
...
```

## Data Models

### StockData
Represents raw data fetched from Yahoo Finance.

```python
@dataclass
class StockData:
    symbol: str                          # Stock symbol (e.g., 'RELIANCE.NS')
    current_price: float                 # Most recent closing price
    historical_prices: List[float]       # Last 30 days of closing prices
    fetch_success: bool                  # Whether data fetch succeeded
    error_message: Optional[str] = None  # Error details if fetch failed
```

### AnalysisResult
Represents analyzed stock with calculated metrics.

```python
@dataclass
class AnalysisResult:
    symbol: str                              # Stock symbol
    current_price: float                     # Current trading price
    moving_average_20d: Optional[float]      # 20-day moving average (None if insufficient data)
    percentage_difference: Optional[float]   # Percentage above/below MA
    is_highlighted: bool                     # True if >= 5% above MA
    error_message: Optional[str] = None      # Error details if analysis failed
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Error isolation
*For any* list of stock symbols containing both valid and invalid symbols, processing failures for invalid symbols should not prevent the successful processing of valid symbols, and all failed symbols should be marked with error information.
**Validates: Requirements 1.4, 6.2**

### Property 2: Current price inclusion
*For any* stock that is successfully fetched, the resulting StockData object should contain a valid current_price value (non-negative float).
**Validates: Requirements 1.5**

### Property 3: Sufficient historical data
*For any* stock that is successfully fetched, the historical_prices list should contain at least 20 data points to enable moving average calculation.
**Validates: Requirements 2.1**

### Property 4: Moving average correctness
*For any* list of 20 or more closing prices, the calculated 20-day moving average should equal the arithmetic mean of the most recent 20 prices: sum(prices[-20:]) / 20.
**Validates: Requirements 2.2**

### Property 5: Percentage difference formula
*For any* current price and moving average pair, the calculated percentage difference should equal ((current_price - moving_average) / moving_average) * 100.
**Validates: Requirements 3.1, 3.4**

### Property 6: Highlight threshold accuracy
*For any* analyzed stock, is_highlighted should be True if and only if percentage_difference >= 5.0.
**Validates: Requirements 3.2**

### Property 7: CSV column completeness
*For any* exported CSV file, parsing the header row should reveal exactly these columns in order: Symbol, Current Price, 20-Day MA, Percentage Difference, Highlighted.
**Validates: Requirements 4.2**

### Property 8: Filename timestamp format
*For any* auto-generated CSV filename, it should match the pattern "nifty50_analysis_YYYYMMDD_HHMMSS.csv" where the timestamp represents a valid date and time.
**Validates: Requirements 4.4**

### Property 9: Failed stock tracking
*For any* analysis execution where some stocks fail to fetch, the final results should include entries for failed stocks with error_message populated and is_highlighted set to False.
**Validates: Requirements 6.4**

## Error Handling

### Network Errors
- **Retry Strategy**: Exponential backoff with 3 retry attempts (delays: 1s, 2s, 4s)
- **Timeout**: 10-second timeout per request to prevent hanging
- **Graceful Degradation**: Failed stocks are logged but don't halt execution

### Data Validation Errors
- **Insufficient Historical Data**: If fewer than 20 days available, set moving_average_20d to None
- **Invalid Price Data**: If current price is negative or zero, mark as error
- **Missing Data**: If yfinance returns empty dataset, mark as fetch failure

### File System Errors
- **Write Permissions**: Check write permissions before attempting CSV export
- **Disk Space**: Basic validation that output directory is writable
- **Path Errors**: Use absolute paths and validate directory existence

### Error Reporting
All errors are captured in the respective data structures (StockData.error_message, AnalysisResult.error_message) and reported in:
1. Console output during execution
2. Final summary statistics
3. CSV output (error stocks included with error indicators)

## Testing Strategy

### Unit Testing Framework
- **Framework**: pytest
- **Coverage Target**: 80% code coverage minimum
- **Test Organization**: Mirror source structure in `tests/` directory

### Unit Tests
Unit tests will cover:
- **Moving Average Calculation**: Test with known price sequences (e.g., [100, 110, 120, ...] should yield predictable MA)
- **Percentage Difference**: Test with edge cases (zero MA, negative differences, exact 5% threshold)
- **CSV Export**: Verify file creation, header format, and data row structure
- **Symbol Provider**: Verify exactly 50 symbols returned with .NS suffix
- **Error Handling**: Test retry logic with mocked network failures

### Property-Based Testing Framework
- **Framework**: Hypothesis (Python property-based testing library)
- **Iterations**: Minimum 100 test cases per property
- **Strategy**: Generate random but valid test data to verify universal properties

### Property-Based Tests
Each property-based test will:
1. Generate random valid inputs (stock prices, symbol lists, etc.)
2. Execute the function under test
3. Verify the correctness property holds
4. Be tagged with format: `# Feature: nifty-stock-analyzer, Property X: [property description]`

Property tests will verify:
- **Property 1**: Error isolation with randomly generated valid/invalid symbol mixes
- **Property 2**: Current price inclusion across random successful fetches
- **Property 3**: Historical data sufficiency across random fetch results
- **Property 4**: Moving average mathematical correctness with random price sequences
- **Property 5**: Percentage formula correctness with random price/MA pairs
- **Property 6**: Highlight threshold accuracy with random percentage differences
- **Property 7**: CSV column structure across random result sets
- **Property 8**: Filename timestamp format across random execution times
- **Property 9**: Failed stock tracking with random failure scenarios

### Integration Testing
- **End-to-End Test**: Run against a small subset of real Nifty 50 symbols (3-5 stocks)
- **Mock Testing**: Use mocked yfinance responses for predictable testing
- **CSV Validation**: Parse generated CSV files to verify data integrity

### Test Data Strategy
- **Mock Data**: Create realistic StockData and AnalysisResult fixtures
- **Price Generators**: Generate valid price sequences (positive floats with realistic ranges)
- **Symbol Generators**: Generate valid NSE symbol formats
- **Edge Cases**: Empty lists, single-element lists, exactly 20 elements, boundary values

## Dependencies

### External Libraries
```
yfinance>=0.2.28        # Yahoo Finance data retrieval
pandas>=2.0.0           # Data manipulation (used by yfinance)
hypothesis>=6.90.0      # Property-based testing
pytest>=7.4.0           # Unit testing framework
```

### Python Version
- **Minimum**: Python 3.9
- **Recommended**: Python 3.11+

## Performance Considerations

### Expected Performance
- **Data Fetching**: ~2-3 seconds per stock (network dependent)
- **Total Execution**: ~2-3 minutes for all 50 stocks
- **Memory Usage**: < 100MB for typical execution

### Optimization Strategies
- **Sequential Processing**: Avoid rate limiting from Yahoo Finance
- **Minimal Data Storage**: Keep only necessary historical data (30 days)
- **Efficient CSV Writing**: Use pandas for optimized CSV export

### Scalability
The current design is optimized for the fixed set of 50 Nifty stocks. For larger datasets:
- Consider parallel fetching with rate limiting
- Implement caching for historical data
- Use database storage instead of CSV for large result sets

## Future Enhancements

Potential future features (out of scope for initial implementation):
1. **Configurable Thresholds**: Allow users to specify custom percentage thresholds
2. **Multiple Moving Averages**: Support 50-day, 100-day, 200-day MAs
3. **Technical Indicators**: Add RSI, MACD, Bollinger Bands
4. **Historical Analysis**: Compare current signals against historical patterns
5. **Alerts**: Email/SMS notifications when stocks meet criteria
6. **Web Dashboard**: Interactive visualization of results
7. **Database Storage**: Persist historical analysis results for trend tracking
