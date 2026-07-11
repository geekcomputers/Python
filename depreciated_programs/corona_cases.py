#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corona_cases.py – COVID-19 Data Fetcher (using disease.sh API)

This script retrieves real-time global and country-specific COVID-19 statistics
from the public disease.sh API (https://disease.sh) and displays them
in a user-friendly terminal interface with ASCII art.

API Endpoints:
    - Global: https://disease.sh/v3/covid-19/all
    - India:  https://disease.sh/v3/covid-19/countries/India

Usage:
    python corona_cases.py

Interactions:
    - Enter '1' to view global statistics.
    - Enter '2' to view statistics for India.
    - Any other input will prompt for re-entry.
"""

import sys
import time

import requests

# API endpoints
GLOBAL_API = "https://disease.sh/v3/covid-19/all"
INDIA_API = "https://disease.sh/v3/covid-19/countries/India"
TIMEOUT = 10  # seconds
MAX_RETRIES = 3  # number of attempts before giving up


def fetch_data(url, description="data"):
    """
    Fetch COVID-19 data from a given URL with retries.

    Args:
        url (str): The API endpoint.
        description (str): A human-readable description for logging.

    Returns:
        dict: Parsed JSON response.

    Raises:
        SystemExit: If all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} to fetch {description} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = 2**attempt
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"❌ Unable to fetch {description} after multiple attempts.")
                sys.exit(1)


def format_world_stats(data):
    """
    Format global statistics into a human-readable string with ASCII art.

    Args:
        data (dict): Global data from disease.sh API.

    Returns:
        str: Formatted string containing global stats.
    """
    ascii_art = """
▀▀█▀▀ █▀▀█ ▀▀█▀▀ █▀▀█ █░░ 　 ▒█▀▀█ █▀▀█ █▀▀ █▀▀ █▀▀ 　 ▀█▀ █▀▀▄ 　 ▒█░░▒█ █▀▀█ █▀▀█ █░░ █▀▀▄ 
░▒█░░ █░░█ ░░█░░ █▄▄█ █░░ 　 ▒█░░░ █▄▄█ ▀▀█ █▀▀ ▀▀█ 　 ▒█░ █░░█ 　 ▒█▒█▒█ █░░█ █▄▄▀ █░░ █░░█ 
░▒█░░ ▀▀▀▀ ░░▀░░ ▀░░▀ ▀▀▀ 　 ▒█▄▄█ ▀░░▀ ▀▀▀ ▀▀▀ ▀▀▀ 　 ▄█▄ ▀░░▀ 　 ▒█▄▀▄█ ▀▀▀▀ ▀░▀▀ ▀▀▀ ▀▀▀░
"""
    stats = (
        f"New Confirmed Cases :- {data.get('todayCases', 0)}\n"
        f"Total Confirmed Cases :- {data.get('cases', 0)}\n"
        f"New Deaths :- {data.get('todayDeaths', 0)}\n"
        f"Total Deaths :- {data.get('deaths', 0)}\n"
        f"New Recovered :- {data.get('todayRecovered', 0)}\n"
        f"Total Recovered :- {data.get('recovered', 0)}\n"
        f"Active Cases :- {data.get('active', 0)}\n"
        f"Critical Cases :- {data.get('critical', 0)}"
    )
    return f"{ascii_art}\n{stats}"


def format_india_stats(data):
    """
    Format India's statistics into a human-readable string with ASCII art.

    Args:
        data (dict): India data from disease.sh API.

    Returns:
        str: Formatted string containing India's stats.
    """
    ascii_art = """
██╗███╗░░██╗██████╗░██╗░█████╗░
██║████╗░██║██╔══██╗██║██╔══██╗
██║██╔██╗██║██║░░██║██║███████║
██║██║╚████║██║░░██║██║██╔══██║
██║██║░╚███║██████╔╝██║██║░░██║
╚═╝╚═╝░░╚══╝╚═════╝░╚═╝╚═╝░░╚═╝
"""
    stats = (
        f"Country Name :- {data.get('country', 'India')}\n"
        f"New Confirmed Cases :- {data.get('todayCases', 0)}\n"
        f"Total Confirmed Cases :- {data.get('cases', 0)}\n"
        f"New Deaths :- {data.get('todayDeaths', 0)}\n"
        f"Total Deaths :- {data.get('deaths', 0)}\n"
        f"New Recovered :- {data.get('todayRecovered', 0)}\n"
        f"Total Recovered :- {data.get('recovered', 0)}\n"
        f"Active Cases :- {data.get('active', 0)}\n"
        f"Critical Cases :- {data.get('critical', 0)}\n"
        f"Information Till :- {data.get('updated', '')}"
    )
    return f"{ascii_art}\n{stats}"


def main():
    """
    Main interactive loop: fetch data, display menu, and show selected stats.
    """
    print("🌐 Fetching latest COVID-19 data...")
    global_data = fetch_data(GLOBAL_API, "global data")
    india_data = fetch_data(INDIA_API, "India data")

    # Print the big title
    title_art = """
░█████╗░░█████╗░██████╗░░█████╗░███╗░░██╗░█████╗░  ██╗░░░██╗██╗██████╗░██╗░░░██╗░██████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗░██║██╔══██╗  ██║░░░██║██║██╔══██╗██║░░░██║██╔════╝
██║░░╚═╝██║░░██║██████╔╝██║░░██║██╔██╗██║███████║  ╚██╗░██╔╝██║██████╔╝██║░░░██║╚█████╗░
██║░░██╗██║░░██║██╔══██╗██║░░██║██║╚████║██╔══██║  ░╚████╔╝░██║██╔══██╗██║░░░██║░╚═══██╗
╚█████╔╝╚█████╔╝██║░░██║╚█████╔╝██║░╚███║██║░░██║  ░░╚██╔╝░░██║██║░░██║╚██████╔╝██████╔╝
░╚════╝░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚══╝╚═╝░░╚═╝  ░░░╚═╝░░░╚═╝╚═╝░░╚═╝░╚═════╝░╚═════╝░
"""
    print(title_art)
    print("\nDeveloped By @TheDarkW3b")
    print("Data source: disease.sh API")

    while True:
        print("\n1 - To Know Corona Virus Update Across World")
        print("2 - To Know Corona Virus Update In India")
        choice = input("Enter 1 Or 2 (or 'q' to quit): ").strip()

        if choice == "1":
            print(format_world_stats(global_data))
            time.sleep(1)
        elif choice == "2":
            print(format_india_stats(india_data))
            time.sleep(1)
        elif choice.lower() == "q":
            print("Exiting... Stay safe!")
            break
        else:
            print("\n⚠️  Invalid input. Please enter 1, 2, or 'q' to quit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
