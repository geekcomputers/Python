import csv
from datetime import datetime

import requests
from bs4 import BeautifulSoup

city = input('Enter City')
url = 'https://www.wunderground.com/weather/in/' + city

try:
    response = requests.get(url)
except requests.exceptions.RequestException as e:
    print(e)
    exit(1)

try:
    response.raise_for_status()
except Exception as e:
    print(e)
    exit(1)

html = response.text
soup = BeautifulSoup(html, 'lxml')
out2 = soup.find(class_="small-12 medium-4 large-3 columns forecast-wrap")
out3 = out2.find(class_="columns small-12")
out4 = soup.find(class_="data-module additional-conditions")

Time = datetime.now().strftime("%H:%M")
Date = out2.find('span', attrs={'class': 'date'}).get_text()
Temperature = out2.find('span', attrs={'class': 'temp'}).get_text()
Temperature = " ".join(Temperature.split())
Precipitation = 'Precipitate:' + out3.find('a', attrs={'class': 'hook'}).get_text().split(' ', 1)[0]
other = out3.find('a', attrs={'class': 'module-link'}).get_text().split('.')
sky = other[0]
Wind = other[2].strip()

with open('weather.csv', 'a') as new_file:
    csv_writer = csv.writer(new_file)
    csv_writer.writerow([city,
                         Time,
                         Date,
                         Temperature,
                         Precipitation,
                         sky, Wind])
