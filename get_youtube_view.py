"""
Created on Thu Apr 27 16:28:36 2017
@author: barnabysandeford
"""
# Currently works for Safari, but just change to whichever 
# browser you're using

import time
from selenium import webdriver

count = int(raw_input("Number of times to be repeated: "))
x = raw_input("Enter the URL (no https): ")
print( "Length of video:")
minutes = int(raw_input("Minutes "))
seconds  = int(raw_input("Seconds "))

refreshrate = minutes * 60 + seconds
driver = webdriver.Safari()
driver.get("http://"+x)

for i in range(count):
    time.sleep(refreshrate)
    driver.refresh()
