'''

Author: Abhinav Anand
git: github.com/ab-anand
mail: abhinavanand1905@gmail.com
Requirements: requests, BeautifulSoup

'''
import os
import webbrowser

import requests
from bs4 import BeautifulSoup
'''
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
'''
query = input('Enter the song to be played: ')
query = query.replace(' ', '+')

# search for the best similar matching video
url = 'https://www.youtube.com/results?search_query=' + query
source_code = requests.get(url,timeout=15)
plain_text = source_code.text
soup = BeautifulSoup(plain_text, "html.parser")

# fetches the url of the video
songs = soup.findAll('div', {'class': 'yt-lockup-video'})
song = songs[0].contents[0].contents[0].contents[0]
link = song['href']
webbrowser.open('https://www.youtube.com' + link)
