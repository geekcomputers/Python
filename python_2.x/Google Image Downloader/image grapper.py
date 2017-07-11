#!/usr/bin/python3

from os import chdir
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from os import walk
import json
from os.path import curdir
from urllib.request import urlretrieve
from os.path import pardir
from create_dir import create_directory

GOOGLE_IMAGE = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
WALLPAPERS_KRAFT = 'https://wallpaperscraft.com/search/keywords?'
usr_agent = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'
        }


def image_grabber(ch):

    # Download images from google images
    if ch == 1:
        print('Enter data to download Images: ')
        data = input()
        search_query = {'q': data}
        search = urlencode(search_query)
        print(search)
        g = GOOGLE_IMAGE + search
        request = Request(g, headers=usr_agent)
        r = urlopen(request).read()
        sew = BeautifulSoup(r, 'html.parser')
        images = []
        # print(sew.prettify())
        results = sew.findAll("div",{"class":"rg_meta"})
        for re in results:
            link, Type = json.loads(re.text)["ou"] , json.loads(re.text)["ity"]
            images.append(link)
        counter = 0
        for re in images:
            rs = requests.get(re)
            with open('img' + str(counter) + '.jpg', 'wb') as file:
                file.write(rs.content)
                # urlretrieve(re, 'img' + str(count) + '.jpg')
                counter += 1
        return True

    elif ch == 2:
        cont = set()  # Stores the links of images
        temp = set()  # Refines the links to download images

        print('Enter data to download wallpapers: ')
        data = input()
        search_query = {'q': data}
        search = urlencode(search_query)
        print(search)
        g = WALLPAPERS_KRAFT + search
        request = Request(g, headers=usr_agent)
        r = urlopen(request).read()
        sew = BeautifulSoup(r, 'html.parser')
        count = 0
        for links in sew.find_all('a'):
            if 'wallpaperscraft.com/download' in links.get('href'):
                cont.add(links.get('href'))
        for re in cont:
            # print all valid links
            # print('https://wallpaperscraft.com/image/' + re[31:-10] + '_' + re[-9:] + '.jpg')
            temp.add('https://wallpaperscraft.com/image/' + re[31:-10] + '_' + re[-9:] + '.jpg')

        # Goes to Each link and downloads high resolution images

        for re in temp:
            rs = requests.get(re)
            with open('img' + str(count) + '.jpg', 'wb') as file:
                file.write(rs.content)
            # urlretrieve(re, 'img' + str(count) + '.jpg')
            count += 1

        return True

    elif ch == 3:
        for folders, subfolder, files in walk(curdir):
            for folder in subfolder:
                print(folder)
        return True

    elif ch == 4:
        print('Enter the directory to be set: ')
        data = input()
        chdir(data + ':\\')
        print('Enter name for the folder: ')
        data = input()
        create_directory(data)
        return True

    elif ch == 5:
        print(
                '''
-------------------------***Thank You For Using***-------------------------
            '''
            )
        return False


run = True

print(
        '''
***********[First Creating Folder To Save Your Images}***********
    '''
    )

create_directory('Images')
DEFAULT_DIRECTORY = pardir + '\\Images'
chdir(DEFAULT_DIRECTORY)

while run:
    print('''
-------------------------WELCOME-------------------------
    1. Search for image
    2. Download Wallpapers 1080p
    3. View Images in your directory
    4. Set directory
    5. Exit
-------------------------*******-------------------------
    ''')
    choice = input()
    run = image_grabber(int(choice))
