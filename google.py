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
    keyword = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else pyperclip.paste()
    res = requests.get(f"http://google.com/search?q={keyword}")
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text)
    linkElems = soup.select(".r a")
    numOpen = min(5, len(linkElems))

    for i in range(numOpen):
        webbrowser.open("http://google.com" + linkElems[i].get("href"))


if __name__ == "__main__":
    main()
