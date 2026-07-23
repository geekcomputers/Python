"""
Time Delta Calculator

This module provides functionality to calculate the absolute difference
in seconds between two timestamps in the format: Day dd Mon yyyy hh:mm:ss +xxxx
"""
# -----------------------------------------------------------------------------
# You are givent two timestams in the format: Day dd Mon yyyy hh:mm:ss +xxxx
# where +xxxx represents the timezone.

# Input Format:
# The first line contains T, the number of test cases.
# Each test case contains two lines, representing the t1 and t2 timestamps.

# Constraints:
# input contains only valid timestamps.
# year is  < 3000.

# Output Format:
# Print the absoulte diffrence (t2 - t1) in seconds.

# Sample Input:
# 2
# Sun 10 May 2015 13:54:36 -0700
# Sun 10 May 2015 13:54:36 -0000
# Sat 02 May 2015 19:54:36 +0530
# Fri 01 May 2015 13:54:36 -0000

# Sample Output:
# 25200
# 88200
# ------------------------------------------------------------------------------

import datetime
from typing import List, Tuple


def parse_timestamp(timestamp: str) -> datetime.datetime:
    """
    Parse a timestamp string into a datetime object.

    Args:
        timestamp: String in the format "Day dd Mon yyyy hh:mm:ss +xxxx"

    Returns:
        A datetime object with timezone information
    """
    # Define the format string to match the input timestamp format
    format_str = "%a %d %b %Y %H:%M:%S %z"
    return datetime.datetime.strptime(timestamp, format_str)


def calculate_time_delta(t1: str, t2: str) -> int:
    """
    Calculate the absolute time difference between two timestamps in seconds.

    Args:
        t1: First timestamp string
        t2: Second timestamp string

    Returns:
        Absolute time difference in seconds as an integer
    """
    # Parse both timestamps
    dt1 = parse_timestamp(t1)
    dt2 = parse_timestamp(t2)

    # Calculate absolute difference and convert to seconds
    time_difference = abs(dt1 - dt2)
    return int(time_difference.total_seconds())


def read_test_cases() -> Tuple[int, List[Tuple[str, str]]]:
    """
    Read test cases from standard input.

    Returns:
        A tuple containing:
        - Number of test cases
        - List of timestamp pairs for each test case
    """
    try:
        num_test_cases = int(input().strip())
        test_cases = []

        for _ in range(num_test_cases):
            timestamp1 = input().strip()
            timestamp2 = input().strip()
            test_cases.append((timestamp1, timestamp2))

        return num_test_cases, test_cases
    except ValueError as e:
        raise ValueError("Invalid input format") from e


def main() -> None:
    """
    Main function to execute the time delta calculation program.
    """
    try:
        num_test_cases, test_cases = read_test_cases()

        for t1, t2 in test_cases:
            result = calculate_time_delta(t1, t2)
            print(result)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
