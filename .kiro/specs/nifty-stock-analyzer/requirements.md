# Requirements Document

## Introduction

This document specifies the requirements for a Python CLI tool that analyzes Nifty 50 stocks in real-time. The tool fetches current stock data using the yfinance library, calculates 20-day moving averages, identifies stocks trading significantly above their moving average, and exports results to CSV format for further analysis.

## Glossary

- **Nifty 50**: The National Stock Exchange of India's benchmark stock market index representing the weighted average of 50 of the largest Indian companies listed on the exchange
- **CLI Tool**: Command-Line Interface Tool - a program that operates through text-based commands in a terminal
- **yfinance**: A Python library that provides access to Yahoo Finance market data
- **20-day Moving Average**: The arithmetic mean of a stock's closing prices over the previous 20 trading days
- **Trading Above Average**: When a stock's current price exceeds its moving average by a specified percentage threshold
- **Stock Analyzer**: The system being developed

## Requirements

### Requirement 1

**User Story:** As a stock trader, I want to fetch real-time data for all Nifty 50 stocks, so that I can analyze current market conditions.

#### Acceptance Criteria

1. WHEN the Stock Analyzer is executed, THE Stock Analyzer SHALL retrieve the list of all 50 Nifty 50 stock symbols
2. WHEN retrieving stock data, THE Stock Analyzer SHALL fetch current price data using the yfinance library
3. WHEN fetching data for multiple stocks, THE Stock Analyzer SHALL handle each stock symbol sequentially to ensure data completeness
4. IF a stock symbol fails to retrieve data, THEN THE Stock Analyzer SHALL log the error and continue processing remaining stocks
5. WHEN all data is retrieved, THE Stock Analyzer SHALL include the current trading price for each stock

### Requirement 2

**User Story:** As a technical analyst, I want the tool to calculate the 20-day moving average for each stock, so that I can identify price trends.

#### Acceptance Criteria

1. WHEN historical data is available, THE Stock Analyzer SHALL retrieve at least 20 days of closing price data for each stock
2. WHEN calculating the moving average, THE Stock Analyzer SHALL compute the arithmetic mean of the most recent 20 closing prices
3. IF fewer than 20 days of data are available for a stock, THEN THE Stock Analyzer SHALL mark the moving average as unavailable and continue processing
4. WHEN the calculation completes, THE Stock Analyzer SHALL store both the current price and the 20-day moving average for each stock

### Requirement 3

**User Story:** As an investor, I want to identify stocks trading 5% above their 20-day moving average, so that I can spot potential momentum opportunities.

#### Acceptance Criteria

1. WHEN comparing prices to moving averages, THE Stock Analyzer SHALL calculate the percentage difference between current price and 20-day moving average
2. WHEN a stock's current price exceeds its 20-day moving average by 5% or more, THE Stock Analyzer SHALL flag that stock as highlighted
3. WHEN displaying results, THE Stock Analyzer SHALL clearly indicate which stocks meet the 5% threshold criteria
4. WHEN calculating percentage differences, THE Stock Analyzer SHALL use the formula: ((current_price - moving_average) / moving_average) * 100

### Requirement 4

**User Story:** As a data analyst, I want the results saved to a CSV file, so that I can perform further analysis in spreadsheet applications.

#### Acceptance Criteria

1. WHEN analysis is complete, THE Stock Analyzer SHALL create a CSV file containing all analyzed stock data
2. WHEN writing to CSV, THE Stock Analyzer SHALL include columns for stock symbol, current price, 20-day moving average, percentage difference, and highlight status
3. WHEN the CSV file already exists, THE Stock Analyzer SHALL overwrite it with new data
4. WHEN the CSV is created, THE Stock Analyzer SHALL use a descriptive filename that includes a timestamp
5. WHEN writing completes, THE Stock Analyzer SHALL display the output file path to the user

### Requirement 5

**User Story:** As a CLI user, I want clear feedback during execution, so that I understand what the tool is doing and when it completes.

#### Acceptance Criteria

1. WHEN the Stock Analyzer starts, THE Stock Analyzer SHALL display a message indicating the analysis has begun
2. WHILE fetching data, THE Stock Analyzer SHALL display progress indicators showing which stocks are being processed
3. WHEN errors occur, THE Stock Analyzer SHALL display clear error messages with details about what failed
4. WHEN analysis completes successfully, THE Stock Analyzer SHALL display a summary including the number of stocks analyzed and the number meeting the highlight criteria
5. WHEN the tool finishes, THE Stock Analyzer SHALL display the location of the output CSV file

### Requirement 6

**User Story:** As a developer, I want the tool to handle network failures gracefully, so that temporary connectivity issues don't crash the application.

#### Acceptance Criteria

1. IF network connectivity is lost during data retrieval, THEN THE Stock Analyzer SHALL retry the request up to 3 times with exponential backoff
2. IF all retry attempts fail for a stock, THEN THE Stock Analyzer SHALL log the failure and continue with remaining stocks
3. WHEN network errors occur, THE Stock Analyzer SHALL display user-friendly error messages rather than technical stack traces
4. WHEN the analysis completes with some failures, THE Stock Analyzer SHALL report which stocks could not be processed
