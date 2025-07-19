import os

import psutil
from obs import watcher

# List of browser process names to monitor
BROWSERS: list[str] = ["chrome.exe", "firefox.exe", "edge.exe", "iexplore.exe"]


def get_downloads_path() -> str:
    """Automatically detect the user's Downloads folder path"""
    return os.path.join(os.path.expanduser("~"), "Downloads")


# Path to monitor (default to user's Downloads folder)
path_to_watch: str = get_downloads_path()


def is_browser_running() -> bool:
    """Check if any browser from the list is running"""
    running_processes = [p.name() for p in psutil.process_iter()]
    return any(browser in running_processes for browser in BROWSERS)


def start_watcher() -> None:
    """Start monitoring the specified folder"""
    if not os.path.exists(path_to_watch):
        raise FileNotFoundError(f"Monitoring path does not exist: {path_to_watch}")

    print(f"Monitoring folder: {path_to_watch}")
    watcher(path_to_watch)


if __name__ == "__main__":
    try:
        # Continuously monitor while any browser is running
        while is_browser_running():
            start_watcher()
        print("No browsers detected. Exiting program.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
