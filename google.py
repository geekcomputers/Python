"""
Author: Ankit Agarwal (ankit167)
Usage: python google.py <keyword>
Description: Script googles the keyword and opens
             top 5 (max) search results in separate
             tabs in the browser
Version: 1.0
"""

import sys
import webbrowser

import bs4
import pyperclip
import requests


def main():
    if len(sys.argv) > 1:
        keyword = ' '.join(sys.argv[1:])
    else:
        # if no keyword is entered, the script would search for the keyword
        # copied in the clipboard
        keyword = pyperclip.paste()

    res = requests.get('http://google.com/search?q=' + keyword)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text)
    linkElems = soup.select('.r a')
    numOpen = min(5, len(linkElems))

    for i in range(numOpen):
        webbrowser.open('http://google.com' + linkElems[i].get('href'))


if __name__ == '__main__':
    main()
