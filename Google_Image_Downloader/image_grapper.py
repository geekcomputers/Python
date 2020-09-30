#!/usr/bin/python
# -*- coding: utf-8 -*-
#importing required libraries
import json
from os import chdir, system
from os import walk
from os.path import curdir
from os.path import pardir
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import requests
import ssl
from bs4 import BeautifulSoup
from create_dir import create_directory


ssl._create_default_https_context = ssl._create_unverified_context

GOOGLE_IMAGE = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
WALLPAPERS_KRAFT = 'https://wallpaperscraft.com/search/keywords?'
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

FX = {
    1: 'search_for_image',
    2: 'download_wallpapers_1080p',
    3: 'view_images_directory',
    4: 'set_directory',
    5: 'quit',
}


# Download images from google images

def search_for_image():
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

    results = sew.findAll('div', {'class': 'rg_meta'})
    for re in results:
        (link, Type) = (json.loads(re.text)['ou'],
                        json.loads(re.text)['ity'])
        images.append(link)
    counter = 0
    for re in images:
        rs = requests.get(re)
        with open('img' + str(counter) + '.jpg', 'wb') as file:
            file.write(rs.content)

            # urlretrieve(re, 'img' + str(count) + '.jpg')

            counter += 1
    return True


def download_wallpapers_1080p():
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

        temp.add('https://wallpaperscraft.com/image/' + re[31:-10] + '_'
                 + re[-9:] + '.jpg')

    # Goes to Each link and downloads high resolution images

    for re in temp:
        rs = requests.get(re)
        with open('img' + str(count) + '.jpg', 'wb') as file:
            file.write(rs.content)

        # urlretrieve(re, 'img' + str(count) + '.jpg')

        count += 1

    return True


###################
def view_images_directory():
    for (folders, subfolder, files) in walk(curdir):
        for folder in subfolder:
            print(folder)
    return True


#############
def set_directory():
    print('Enter the directory to be set: ')
    data = input()
    chdir(data + ':\\')
    print('Enter name for the folder: ')
    data = input()
    create_directory(data)
    return True


##############
def quit():
    print('''
-------------------------***Thank You For Using***-------------------------
        ''')
    return False


run = True

print('''
***********[First Creating Folder To Save Your Images}***********
    ''')

create_directory('Images')
DEFAULT_DIRECTORY = pardir + '\\Images'
chdir(DEFAULT_DIRECTORY)
count = 0
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
    try:
        # Via eval() let `str expression` to `function`
        fx = eval(FX[int(choice)])
        run = fx()
    except KeyError:
        system('clear')
        if count <= 5:
            count += 1
            print("----------enter proper key-------------")
        else:
            system('clear')
            print("You have attempted 5 times , try again later")
            run = False
