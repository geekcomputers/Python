import urllib.request

import mechanize
from bs4 import BeautifulSoup

# Create a Browser
browser = mechanize.Browser()

# Disable loading robots.txt
browser.set_handle_robots(False)

browser.addheaders = [("User-agent", "Mozilla/4.0 (compatible; MSIE 5.0; Windows 98;)")]

movie_title = input("Enter movie title: ")

movie_types = (
    "feature",
    "tv_movie",
    "tv_series",
    "tv_episode",
    "tv_special",
    "tv_miniseries",
    "documentary",
    "video_game",
    "short",
    "video",
    "tv_short",
)

# Navigate
browser.open("http://www.imdb.com/search/title")

# Choose a form
browser.select_form(nr=1)

browser["title"] = movie_title

# Check all the boxes of movie types
for m_type in movie_types:
    browser.find_control(type="checkbox", nr=0).get(m_type).selected = True

# Submit
fd = browser.submit()
soup = BeautifulSoup(fd.read(), "html5lib")

# Updated from td tag to h3 tag
for div in soup.findAll("h3", {"class": "lister-item-header"}, limit=1):
    a = div.findAll("a")[0]
    hht = "http://www.imdb.com" + a.attrs["href"]
    print(hht)
    page = urllib.request.urlopen(hht)
    soup2 = BeautifulSoup(page.read(), "html.parser")
    find = soup2.find

    print("Title: " + find(itemprop="name").get_text().strip())
    print("Duration: " + find(itemprop="duration").get_text().strip())
    print("Director: " + find(itemprop="director").get_text().strip())
    print("Genre: " + find(itemprop="genre").get_text().strip())
    print("IMDB rating: " + find(itemprop="ratingValue").get_text().strip())
    print("Summary: " + find(itemprop="description").get_text().strip())
