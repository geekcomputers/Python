#!/usr/bin/env python3

import sys
from urllib.error import URLError
from urllib.request import urlopen


def check_internet_connectivity() -> None:
    """
    Check internet connectivity by attempting to reach a specified URL.
    Uses google.com as fallback if no URL is provided.
    """
    try:
        # Get URL from command-line argument or use default
        url = sys.argv[1] if len(sys.argv) > 1 else "https://google.com"

        # Ensure URL starts with a protocol
        if not any(url.startswith(p) for p in ["https://", "http://"]):
            url = "https://" + url

        print(f"Checking URL: {url}")

        # Attempt connection with 2-second timeout
        urlopen(url, timeout=2)
        print(f'Connection to "{url}" is working')

    except URLError as e:
        print(f"Connection error: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    check_internet_connectivity()
