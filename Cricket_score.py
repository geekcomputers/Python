from urllib import request

import bs4  # Beautiful Soup for Web Scraping
from win10toast import ToastNotifier

toaster = ToastNotifier()

url = "http://www.cricbuzz.com/cricket-match/live-scores"

sauce = request.urlopen(url).read()
soup = bs4.BeautifulSoup(sauce, "lxml")

score = []
results = []

for div_tags in soup.find_all('div', attrs={"class": "cb-lv-scrs-col text-black"}):
    score.append(div_tags.text)
for result in soup.find_all('div', attrs={"class": "cb-lv-scrs-col cb-text-complete"}):
    results.append(result.text)

print(score[0], results[0])
toaster.show_toast(title=score[0], msg=results[0])
