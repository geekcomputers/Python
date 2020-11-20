"""
Written by: Shreyas Daniel - github.com/shreydan
Written on: 26 April 2017

Description: Download latest XKCD Comic with this program.

NOTE:
	if this script is launched from the cloned repo, a new folder is created.
	Please move the file to another directory to avoid messing with the folder structure.
"""

import os
import urllib.request
import json
import requests


def main():
    # opens xkcd.com
    try:
        page = requests.get("https://www.xkcd.com/info.0.json")
    except requests.exceptions.RequestException as e:
        print(e)
        exit()
    data = json.loads(page.text)
    # save location of comic
    image_src = data['img']
    comic_location = os.getcwd() + '/comics/'

    # checks if save location exists else creates
    if not os.path.exists(comic_location):
        os.makedirs(comic_location)

    # creates final comic location including name of the comic
    comic_location = comic_location + comic_name

    # downloads the comic
    urllib.request.urlretrieve(image_src, comic_location)


if __name__ == "__main__":
    main()
