import bs4
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq

my_url="http://www.cricbuzz.com/"
Client=uReq(my_url)

html_page=Client.read()
Client.close()


soup_page=soup(html_page,"html.parser")


score_box=soup_page.findAll("div",{"class":"cb-col cb-col-25 cb-mtch-blk"})
l = len(score_box)
print(l)
for i in range(l):
	print(score_box[i].a["title"])
	print(score_box[i].a.text)
	print()
