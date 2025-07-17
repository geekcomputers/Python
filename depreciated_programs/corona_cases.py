import sys
from typing import Dict, List, Any, Optional
import httpx
from time import sleep

# Type aliases
CovidData = Dict[str, Any]
CountryData = List[Dict[str, Any]]

# API request and data processing
url: str = "https://api.covid19api.com/summary"

try:
    response: httpx.Response = httpx.get(url)
    response.raise_for_status()  # Check if request was successful
    visit: CovidData = response.json()
except ImportError:
    print("Please install the HTTPX module using 'pip install httpx'")
    sys.exit(1)
except (httpx.RequestError, ValueError) as e:
    print(f"Failed to fetch data: {e}")
    sys.exit(1)

# Extract global data
global_data: Dict[str, int] = visit["Global"]
NewConfirmed: int = global_data["NewConfirmed"]
TotalConfirmed: int = global_data["TotalConfirmed"]
NewDeaths: int = global_data["NewDeaths"]
TotalDeaths: int = global_data["TotalDeaths"]
NewRecovered: int = global_data["NewRecovered"]
TotalRecovered: int = global_data["TotalRecovered"]

# Extract India data (using country name instead of index for reliability)
countries: CountryData = visit["Countries"]
india_data: Optional[Dict[str, Any]] = next(
    (country for country in countries if country["Country"] == "India"),
    None
)

if india_data is None:
    print("Error: India's data not found in the API response.")
    sys.exit(1)

name: str = india_data["Country"]
indiaconfirmed: int = india_data["NewConfirmed"]
indiatotal: int = india_data["TotalConfirmed"]
indiaDeaths: int = india_data["NewDeaths"]
deathstotal: int = india_data["TotalDeaths"]
indianewr: int = india_data["NewRecovered"]
totalre: int = india_data["TotalRecovered"]
DateUpdate: str = india_data["Date"]

def world() -> None:
    """Display global COVID-19 statistics"""
    world_stats = f"""
▀▀█▀▀ █▀▀█ ▀▀█▀▀ █▀▀█ █░░ 　 ▒█▀▀█ █▀▀█ █▀▀ █▀▀ █▀▀ 　 ▀█▀ █▀▀▄ 　 ▒█░░▒█ █▀▀█ █▀▀█ █░░ █▀▀▄ 
░▒█░░ █░░█ ░░█░░ █▄▄█ █░░ 　 ▒█░░░ █▄▄█ ▀▀█ █▀▀ ▀▀█ 　 ▒█░ █░░█ 　 ▒█▒█▒█ █░░█ █▄▄▀ █░░ █░░█ 
░▒█░░ ▀▀▀▀ ░░▀░░ ▀░░▀ ▀▀▀ 　 ▒█▄▄█ ▀░░▀ ▀▀▀ ▀▀▀ ▀▀▀ 　 ▄█▄ ▀░░▀ 　 ▒█▄▀▄█ ▀▀▀▀ ▀░▀▀ ▀▀▀ ▀▀▀░\n
New Confirmed Cases: {NewConfirmed}
Total Confirmed Cases: {TotalConfirmed}
New Deaths: {NewDeaths}
Total Deaths: {TotalDeaths}
New Recovered: {NewRecovered}
Total Recovered: {TotalRecovered}
    """
    print(world_stats)

def india() -> None:
    """Display COVID-19 statistics for India"""
    india_stats = f"""
██╗███╗░░██╗██████╗░██╗░█████╗░
██║████╗░██║██╔══██╗██║██╔══██╗
██║██╔██╗██║██║░░██║██║███████║
██║██║╚████║██║░░██║██║██╔══██║
██║██║░╚███║██████╔╝██║██║░░██║
╚═╝╚═╝░░╚══╝╚═════╝░╚═╝╚═╝░░╚═╝

Country: {name}
New Confirmed Cases: {indiaconfirmed}
Total Confirmed Cases: {indiatotal}
New Deaths: {indiaDeaths}
Total Deaths: {deathstotal}
New Recovered: {indianewr}
Total Recovered: {totalre}
Updated As Of: {DateUpdate}
"""
    print(india_stats)

# ASCII art title
print("""
░█████╗░░█████╗░██████╗░░█████╗░███╗░░██╗░█████╗░  ██╗░░░██╗██╗██████╗░██╗░░░██╗░██████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗░██║██╔══██╗  ██║░░░██║██║██╔══██╗██║░░░██║██╔════╝
██║░░╚═╝██║░░██║██████╔╝██║░░██║██╔██╗██║███████║  ╚██╗░██╔╝██║██████╔╝██║░░░██║╚█████╗░
██║░░██╗██║░░██║██╔══██╗██║░░██║██║╚████║██╔══██║  ░╚████╔╝░██║██╔══██╗██║░░░██║░╚═══██╗
╚█████╔╝╚█████╔╝██║░░██║╚█████╔╝██║░╚███║██║░░██║  ░░╚██╔╝░░██║██║░░██║╚██████╔╝██████╔╝
░╚════╝░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚══╝╚═╝░░╚═╝  ░░░╚═╝░░░╚═╝╚═╝░░╚═╝░╚═════╝░╚═════╝░""")
print("\nDeveloped By @TheDarkW3b")

def choices() -> None:
    """Main menu for user choices"""
    print("\n1 - View Global COVID-19 Updates")
    print("\n2 - View COVID-19 Updates in India")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        world()
        sleep(1)
        choices()
    elif choice == "2":
        india()
        sleep(1)
        choices()
    else:
        print("\nInvalid input. Please try again.")
        choices()

# Start the interactive menu
choices()