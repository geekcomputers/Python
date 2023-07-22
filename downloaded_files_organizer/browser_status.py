import psutil
from obs import watcher

browsers = ["chrome.exe", "firefox.exe", "edge.exe", "iexplore.exe"]

# ADD DOWNLOADS PATH HERE::: r is for raw string enter the path
# Example: path_to_watch=r"C:\Users\Xyz\Downloads"
# find downloads path .


path_to_watch = r" "


for browser in browsers:
    while browser in (process.name() for process in psutil.process_iter()):
        watcher(path_to_watch)
