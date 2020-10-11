'''
Author: Anshu Saini
GitHub: https://github.com/anshu189
mail: anshusaini189381@gmail.com
Requirements: Selenium (pip install selenium), webdriver (https://sites.google.com/a/chromium.org/chromedriver/downloads)
'''

from selenium import webdriver
from time import sleep as s

song_name = input("Enter your song name: ")
artist_name = input("Enter the artist name(optional): ")
add = song_name + artist_name

link = "https://www.youtube.com/results?search_query=" + add

driver_path = "C:/chromedriver.exe"  # Your Chromedriver.exe path here

#  <---For Brave Browser--->
# brave_path = "C:/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe"  # Your Brave.exe path here
# option = webdriver.ChromeOptions()
# option.binary_location = brave_path

# driv = webdriver.Chrome(executable_path=driver_path, options=option)
driv = webdriver.Chrome(driver_path)

driv.maximize_window()
driv.get(link)
s(0.5)
driv.find_element_by_xpath("//*[@id='video-title']").click()
