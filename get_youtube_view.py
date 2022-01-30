"""
Created on Thu Apr 27 16:28:36 2017
@author: barnabysandeford
"""
# Currently works for Safari, but just change to whichever
# browser you're using.

import time

# Added pafy to get video length for the user
import pafy

# Changed the method of opening the browser.
# Selenium allows for the page to be refreshed.
from selenium import webdriver

# adding ability to change number of repeats
count = int(input("Number of times to be repeated: "))
# Same as before
url = input("Enter the URL : ")

refreshrate = None

# tries to get video length using pafy
try:
    video = pafy.new(url)
    if hasattr(video, "length"):
        refreshrate = video.length
# if pafy fails to work, prints out error and asks for video length from the user
except Exception as e:
    print(e)
    print("Length of video:")
    minutes = int(input("Minutes "))
    seconds = int(input("Seconds "))
    # Calculating the refreshrate from the user input
    refreshrate = minutes * 60 + seconds

# Selecting Safari as the browser
driver = webdriver.Safari()

if url.startswith("https://"):
    driver.get(url)
else:
    driver.get("https://" + url)

for i in range(count):
    # Sets the page to refresh at the refreshrate.
    time.sleep(refreshrate)
    driver.refresh()
