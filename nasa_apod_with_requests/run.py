from settings import key
import requests
import os

date = input("Enter date(YYYY-MM-DD): ")
r = requests.get(f"https://api.nasa.gov/planetary/apod?api_key={key}&date={date}")
parsed = r.json()
title = parsed['title']
url = parsed['hdurl']
print(f"{title}: {url}")

img_ = requests.get(url, stream=True)
print(img_.headers)
print(img_.headers["content-type"], img_.headers["content-length"])
content_type = img_.headers["content-type"]

if (img_.status_code == 200 and (content_type == "image/jpeg" or content_type == "image/gif" or content_type == "image/png")):
	ext = img_.headers["content-type"][6:]
	if (not os.path.exists ("img/")):
		os.mkdir("img/")
	path = f"img/apod_{date}.{ext}"
	with open(path, "wb") as f:
		for chunk in img_:
			f.write(chunk)