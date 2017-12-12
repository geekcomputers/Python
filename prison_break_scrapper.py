"""
Scrapper for downloading prison break
series from an open server and putting them in a designated folder.
"""
import requests as req
from bs4 import BeautifulSoup as bs
import os
import subprocess


BASE_URL = 'http://dl.funsaber.net/serial/Prison%20Break/season%20'


def download_files(links, idx):
	for link in links:
		subprocess.call([
			"aria2c",
			"-s",
			"16",
			"-x",
			"16",
			"-d",
			"season"+str(idx),
			link
		])


def main():
	for i in range(1,5):
		r = req.get(BASE_URL+str(i)+'/1080/')
		soup = bs(r.text, 'html.parser')
		link_ = []
		for link in soup.find_all('a'):
			if '.mkv' in link.get('href'):
				link_.append(BASE_URL+str(i)+'/1080/'+link.get('href'))
		if not os.path.exists('season'+str(i)):
			os.makedirs('season'+str(i))
		download_files(link_, i)



if __name__ == '__main__':
	main()