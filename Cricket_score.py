import bs4 as bs
from urllib import request
from  win10toast import ToastNotifier

toaster = ToastNotifier()

url = 'http://www.cricbuzz.com/cricket-match/live-scores'

sauce = request.urlopen(url).read()
soup = bs.BeautifulSoup(sauce,"lxml")
#print(soup)
score = []
results = []
#for live_matches in soup.find_all('div',attrs={"class":"cb-mtch-lst cb-col cb-col-100 cb-tms-itm"}):
for div_tags in soup.find_all('div', attrs={"class": "cb-lv-scrs-col text-black"}):
        score.append(div_tags.text)
for result in soup.find_all('div', attrs={"class": "cb-lv-scrs-col cb-text-complete"}):
          results.append(result.text)


print(score[0],results[0])
toaster.show_toast(title=score[0],msg=results[0])

