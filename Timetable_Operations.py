"""Clock Time Difference Calculator

Calculates the time difference between two 24-hour formatted times (HH:MM:SS)
and returns the result in the same HH:MM:SS format. Handles cases where the
second time is earlier than the first (crosses midnight).
"""

import sys


def parse_time(time_str: str) -> tuple[int, int, int]:
    """Parse a time string in HH:MM:SS format into hours, minutes, seconds.

    Args:
        time_str: Time string in "HH:MM:SS" format.

    Returns:
        Tuple containing (hours, minutes, seconds).

    Raises:
        ValueError: If input format is invalid or values are out of range.
    """
    # Validate string structure
    if not isinstance(time_str, str):
        raise ValueError("Time must be a string")

    if len(time_str) != 8:
        raise ValueError(f"Invalid time length: '{time_str}' (expected 8 characters)")

    if time_str[2] != ":" or time_str[5] != ":":
        raise ValueError(f"Invalid format: '{time_str}' (use HH:MM:SS)")

    # Extract and validate components
    try:
        hours = int(time_str[0:2])
        minutes = int(time_str[3:5])
        seconds = int(time_str[6:8])
    except ValueError as e:
        raise ValueError(f"Non-numeric components in '{time_str}': {e}") from e

    # Validate value ranges
    if not (0 <= hours < 24):
        raise ValueError(f"Hours out of range (0-23): {hours}")
    if not (0 <= minutes < 60):
        raise ValueError(f"Minutes out of range (0-59): {minutes}")
    if not (0 <= seconds < 60):
        raise ValueError(f"Seconds out of range (0-59): {seconds}")

    return hours, minutes, seconds


def time_to_seconds(hours: int, minutes: int, seconds: int) -> int:
    """Convert hours, minutes, seconds to total seconds since midnight.

    Args:
        hours: Hours (0-23)
        minutes: Minutes (0-59)
        seconds: Seconds (0-59)

    Returns:
        Total seconds (0-86399).
    """
    # Double-check input ranges (defensive programming)
    if not (0 <= hours < 24):
        raise ValueError(f"Invalid hours: {hours} (must be 0-23)")
    if not (0 <= minutes < 60):
        raise ValueError(f"Invalid minutes: {minutes} (must be 0-59)")
    if not (0 <= seconds < 60):
        raise ValueError(f"Invalid seconds: {seconds} (must be 0-59)")

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time(total_seconds: int) -> tuple[int, int, int]:
    """Convert total seconds to hours, minutes, seconds.

    Args:
        total_seconds: Total seconds (handles negative values and large numbers).

    Returns:
        Tuple containing (hours, minutes, seconds) normalized to 0-23:59:59.
    """
    # Handle negative values by adding full days until positive
    if total_seconds < 0:
        days_to_add = (-total_seconds // 86400) + 1
        total_seconds += days_to_add * 86400

    # Normalize to within a single day (0-86399 seconds)
    total_seconds %= 86400

    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    seconds = remaining % 60

    return hours, minutes, seconds


def format_time(hours: int, minutes: int, seconds: int) -> str:
    """Format hours, minutes, seconds into HH:MM:SS string.

    Args:
        hours: Hours (0-23)
        minutes: Minutes (0-59)
        seconds: Seconds (0-59)

    Returns:
        Formatted time string with leading zeros where necessary.
    """
    # Final validation before formatting
    if not (0 <= hours < 24):
        raise ValueError(f"Invalid hours for formatting: {hours}")
    if not (0 <= minutes < 60):
        raise ValueError(f"Invalid minutes for formatting: {minutes}")
    if not (0 <= seconds < 60):
        raise ValueError(f"Invalid seconds for formatting: {seconds}")

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_time_difference(t1: str, t2: str) -> str:
    """Calculate the time difference between two times (t2 - t1).

    Args:
        t1: Start time in HH:MM:SS format.
        t2: End time in HH:MM:SS format.

    Returns:
        Time difference in HH:MM:SS format.
    """
    # Validate input isn't empty
    if not t1.strip():
        raise ValueError("Initial time cannot be empty")
    if not t2.strip():
        raise ValueError("Final time cannot be empty")

    # Parse input times with detailed error context
    try:
        h1, m1, s1 = parse_time(t1)
    except ValueError as e:
        raise ValueError(f"Invalid initial time: {e}") from e

    try:
        h2, m2, s2 = parse_time(t2)
    except ValueError as e:
        raise ValueError(f"Invalid final time: {e}") from e

    # Convert to total seconds with range checks
    sec1 = time_to_seconds(h1, m1, s1)
    sec2 = time_to_seconds(h2, m2, s2)

    # Calculate difference (handle midnight crossing)
    diff_seconds = sec2 - sec1

    # Convert back to hours, minutes, seconds with normalization
    h_diff, m_diff, s_diff = seconds_to_time(diff_seconds)

    # Format and return result
    return format_time(h_diff, m_diff, s_diff)


def main() -> None:
    """Main function to handle user input and display results."""
    try:
        t1 = input("Initial schedule (HH:MM:SS): ").strip()
        t2 = input("Final schedule (HH:MM:SS): ").strip()

        result = calculate_time_difference(t1, t2)
        print(f"Final result is: {result}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
