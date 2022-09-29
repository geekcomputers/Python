import requests
import json
import geocoder

g = geocoder.ip("me")

lat = g.latlng[0]

longi = g.latlng[1]
query = input("Enter the query")

url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={str(lat)},{str(longi)}radius=1000"


key = "your_api_key"
r = requests.get(f"{url}query={query}&key={key}")

x = r.json()
y = x["results"]
print(y)
