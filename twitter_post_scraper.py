"""
Twitter tweet scraper and text cleaner.

This module fetches tweets from a given Twitter handle, extracts their text,
and removes URLs, special characters, and common noise patterns.
"""

import re
import urllib.parse
from typing import List

import requests
from bs4 import BeautifulSoup

# Regex for cleaning tweet text:
# - Removes common punctuation: : . !
# - Removes URLs (http, https, or protocol-relative) using a more precise pattern
# - Removes Twitter-specific short links like pic.twitter.com/xxx
# - Removes '&' characters
# All dots are escaped to match literal dots only.
URL_PATTERN = re.compile(
    r"(?:https?://)?(?:[\w\-]+\.)+[\w\-]+(?:/[\w\-./?%&=]*)?"
    r"|pic\.twitter\.com/\w+"
    r"|twitter\.com/\w+"
    r"|&",
    flags=re.IGNORECASE,
)

# Additional noise patterns (non-breaking spaces, zero-width joiners)
NOISE_PATTERN = re.compile(r"[\xa0\u200c…]")


def clean_tweet_text(raw_text: str) -> str:
    """
    Remove URLs, extra spaces, and special characters from a tweet.

    Args:
        raw_text: The raw tweet text as extracted from HTML.

    Returns:
        Cleaned text with URLs and noise removed.
    """
    # Remove URLs and '&' using the compiled pattern
    cleaned = URL_PATTERN.sub("", raw_text)

    # Remove non-breaking spaces, zero-width joiners, and ellipsis
    cleaned = NOISE_PATTERN.sub("", cleaned)

    # Collapse multiple spaces and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def fetch_tweets(handle: str) -> List[str]:
    """
    Fetch tweets from a Twitter profile page and return a list of cleaned texts.

    Args:
        handle: Twitter handle (without '@').

    Returns:
        List of cleaned tweet strings.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the handle is empty or invalid.
    """
    if not handle or not handle.isalnum():  # simple sanity check
        raise ValueError("Twitter handle must be non-empty and alphanumeric.")

    base_url = "https://twitter.com/{}"
    url = base_url.format(handle)

    # Send GET request with a user-agent to avoid blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch tweets: {e}")

    soup = BeautifulSoup(resp.content, "lxml")

    # Locate tweet containers (the class may change; adjust if needed)
    tweet_divs = soup.find_all("div", class_="tweet")

    tweets = []
    for tweet in tweet_divs:
        content = tweet.find("div", class_="content")
        if not content:
            continue
        text_container = content.find("div", class_="js-tweet-text-container")
        if not text_container:
            continue

        raw = text_container.get_text(separator=" ", strip=True)
        cleaned = clean_tweet_text(raw)
        if cleaned:  # avoid empty strings
            tweets.append(cleaned)

    return tweets


def main() -> None:
    """Main entry point: ask for a Twitter handle and print cleaned tweets."""
    print("Enter Twitter handle (without @):")
    handle = input().strip()

    try:
        cleaned_tweets = fetch_tweets(handle)
        print("\nCleaned tweets:")
        for i, tweet in enumerate(cleaned_tweets, 1):
            print(f"{i}. {tweet}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()