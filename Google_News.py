import bs4
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

def news():
	#my_url="https://news.google.com/news/rss"
	my_url="https://news.google.com/news/rss?ned=in&hl=en-IN"
	#To open the Given URL
	Client=urlopen(my_url)

	xml_page=Client.read()
	Client.close()

	soup_page=soup(xml_page,"xml")
	news_list=soup_page.findAll("item")
	
	for news in news_list:
			print(news.title.text)
			print(news.link.text)
			print(news.pubDate.text)
			print("-"*150)
	n=input()



news()	
