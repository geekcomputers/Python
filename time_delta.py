"""Time Delta Calculator

This program calculates the absolute time difference in seconds between two
timestamps. Each timestamp includes a date, time, and timezone offset.
"""

import sys
from datetime import datetime, timedelta


def parse_timestamp(timestamp: str) -> datetime:
    """Parse a timestamp string into a timezone-aware datetime object.

    Args:
        timestamp: String in format "Day dd Mon yyyy hh:mm:ss +xxxx"

    Returns:
        Timezone-aware datetime object representing the timestamp.
    """
    # Format codes:
    # %a - Abbreviated weekday name (e.g., Sun)
    # %d - Day of the month (01-31)
    # %b - Abbreviated month name (e.g., May)
    # %Y - 4-digit year
    # %H - Hour (24-hour clock, 00-23)
    # %M - Minute (00-59)
    # %S - Second (00-59)
    # %z - UTC offset in ±HHMM format (e.g., -0700)
    return datetime.strptime(timestamp, "%a %d %b %Y %H:%M:%S %z")


def time_delta(t1: str, t2: str) -> int:
    """Calculate the absolute difference in seconds between two timestamps.

    Args:
        t1: First timestamp string
        t2: Second timestamp string

    Returns:
        Absolute difference in seconds as an integer.
    """
    dt1 = parse_timestamp(t1)
    dt2 = parse_timestamp(t2)
    delta: timedelta = dt1 - dt2
    return abs(int(delta.total_seconds()))


def get_input() -> tuple[int, list[tuple[str, str]]]:
    """Read and validate input from standard input.

    Returns:
        Tuple containing:
        - Number of test cases
        - List of tuples with pairs of timestamps
    """
    try:
        t = int(input().strip())
        if t < 1:
            raise ValueError("Number of test cases must be positive")

        test_cases = []
        for _ in range(t):
            t1 = input().strip()
            t2 = input().strip()
            test_cases.append((t1, t2))

        return t, test_cases

    except ValueError as e:
        raise SystemExit(f"Input error: {e}")


if __name__ == "__main__":
    try:
        num_tests, test_cases = get_input()
        for t1, t2 in test_cases:
            print(time_delta(t1, t2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
