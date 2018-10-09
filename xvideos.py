import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError
import urllib.request
from urllib.request import urlopen


def scrap_link(url):
    source_code = None
    s_number = 1
    try:
        source_code = requests.get(url)
    except ConnectionError:
        print('Failed to load,Check Your Network')
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'html.parser')
    content = soup.findAll('div', {'class': 'thumb'})
    for link in content:
        link_content = link.find('a')
        if link_content.get('title') is not None:
            print(str(s_number) + '. ' + url + '/download/480/' + link_content.get('href')[-7:]+'.mp4')   # url of fetched_videos
            print(link_content.get('title'))
            s_number += 1
    index_number = input('Enter Serial Number To Download video ')
    print(index_number)
    video_link = url + '/download/360/' + content[int(index_number)-1].find('a').get('href')[-7:] + '.mp4'
    print(video_link)
    download_video(video_link)


def download_video(video_url):
    file_name = 'enjoy.mp4'
    request = urllib.request.Request(video_url, method='GET')
    response = urlopen(request)
    with open(file_name, 'wb') as f:
        f.write(response.read())


scrap_link('https://www.porn.com')
