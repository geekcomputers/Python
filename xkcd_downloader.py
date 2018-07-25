"""
Written by: Shreyas Daniel - github.com/shreydan
Written on: 26 April 2017

Description: Download latest XKCD Comic with this program.

NOTE:
	if this script is launched from the cloned repo, a new folder is created.
	Please move the file to another directory to avoid messing with the folder structure.
"""

import requests
from lxml import html
import urllib.request
import os

def main():
    # opens xkcd.com
    try:
        page = requests.get("https://www.xkcd.com") 
    except requests.exceptions.RequestException as e:
        print (e)
        exit()
    
    # parses xkcd.com page
    tree = html.fromstring(page.content)
    
    # finds image src url
    image_src = tree.xpath(".//*[@id='comic']/img/@src")[0]
    image_src = "https:" + str(image_src)
    
    # gets comic name from the image src url
    comic_name = image_src.split('/')[-1]
    
    # save location of comic
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
