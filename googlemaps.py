import requests
import json
import geocoder

g = geocoder.ip('me')

lat = g.latlng[0]

longi = g.latlng[1]

https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=-33.8670522,151.1957362&radius=500&types=food&name=harbour&key=AIzaSyDqWyJzDN21MlLdS_PHqN2GQe_YCUHz6Q8
def find(query):
    key = "your_api_key"
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + str(lat) + "," + str(longi) + "radius=1000"

    r = requests.get(url + 'query=' + query + '&key=' + key)

    x = r.json()
    y = x['results']
    print(y)
