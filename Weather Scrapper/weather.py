#TODO - refactor & clean code
import csv
import time
from datetime import datetime
from datetime import date
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

#TODO - Add input checking
city = input("City >")
state = input("State >")

url = 'https://www.wunderground.com'

#Supresses warnings and specifies the webdriver to run w/o a GUI
options = Options()
options.headless = True
options.add_argument('log-level=3')
driver = webdriver.Chrome(options=options)

driver.get(url)
#-----------------------------------------------------
# Connected successfully to the site
#Passes the city and state input to the weather sites search box

searchBox = driver.find_element(By.XPATH, '//*[@id="wuSearch"]')
location = city + " " + state

action = ActionChains(driver)
searchBox.send_keys(location)
element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="wuForm"]/search-autocomplete/ul/li[2]/a/span[1]'))
)
searchBox.send_keys(Keys.RETURN)
#-----------------------------------------------------
#Gather weather data
#City - Time - Date - Temperature - Precipitation - Sky - Wind

#waits till the page loads to begin gathering data
precipitationElem = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="inner-content"]/div[3]/div[1]/div/div[3]/div/lib-city-today-forecast/div/div[1]/div/div/div/a[1]'))
)
precipitationElem = driver.find_element(By.XPATH, '//*[@id="inner-content"]/div[3]/div[1]/div/div[3]/div/lib-city-today-forecast/div/div[1]/div/div/div/a[1]')
precip = "Precipitation:" + precipitationElem.text.split()[0]

windAndSkyElem = driver.find_element(By.XPATH, '//*[@id="inner-content"]/div[3]/div[1]/div/div[3]/div/lib-city-today-forecast/div/div[1]/div/div/div/a[2]')
description = windAndSkyElem.text.split(". ")
sky = description[0]
temp = description[1]
wind = description[2]

#Format the date and time
time = datetime.now().strftime("%H:%M")
today = date.today()
date = today.strftime("%b-%d-%Y")

print(city, time, date, temp, precip, sky, wind)

with open("weather.csv", "a") as new_file:
    csv_writer = csv.writer(new_file)
    csv_writer.writerow([city, time, date, temp, precip, sky, wind])

driver.close()
