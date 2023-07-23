import tkinter as tk
import requests
from bs4 import BeautifulSoup

url = "https://weather.com/en-IN/weather/today/l/32355ced66b7ce3ab7ccafb0a4f45f12e7c915bcf8454f712efa57474ba8d6c8"

root = tk.Tk()
root.title("Weather")
root.config(bg="white")


def getWeather():
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    location = soup.find("h1", class_="_1Ayv3").text
    temperature = soup.find("span", class_="_3KcTQ").text
    airquality = soup.find("text", class_="k2Z7I").text
    airqualitytitle = soup.find("span", class_="_1VMr2").text
    sunrise = soup.find("div", class_="_2ATeV").text
    sunset = soup.find("div", class_="_2_gJb _2ATeV").text
    # humidity = soup.find('div',class_='_23DP5').text
    wind = soup.find("span", class_="_1Va1P undefined").text
    pressure = soup.find("span", class_="_3olKd undefined").text
    locationlabel.config(text=(location))
    templabel.config(text=temperature + "C")
    WeatherText = (
        "Sunrise : "
        + sunrise
        + "\n"
        + "SunSet : "
        + sunset
        + "\n"
        + "Pressure : "
        + pressure
        + "\n"
        + "Wind : "
        + wind
        + "\n"
    )
    weatherPrediction.config(text=WeatherText)
    airqualityText = airquality + " " * 5 + airqualitytitle + "\n"
    airqualitylabel.config(text=airqualityText)

    weatherPrediction.after(120000, getWeather)
    root.update()


locationlabel = tk.Label(root, font=("Calibri bold", 20), bg="white")
locationlabel.grid(row=0, column=1, sticky="N", padx=20, pady=40)

templabel = tk.Label(root, font=("Caliber bold", 40), bg="white")
templabel.grid(row=0, column=0, sticky="W", padx=17)

weatherPrediction = tk.Label(root, font=("Caliber", 15), bg="white")
weatherPrediction.grid(row=2, column=1, sticky="W", padx=40)

tk.Label(root, text="Air Quality", font=("Calibri bold", 20), bg="white").grid(
    row=1, column=2, sticky="W", padx=20
)
airqualitylabel = tk.Label(root, font=("Caliber bold", 20), bg="white")
airqualitylabel.grid(row=2, column=2, sticky="W")

getWeather()
root.mainloop()
