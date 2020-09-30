import ssl
from urllib.request import urlopen

from bs4 import BeautifulSoup as soup

def news(xml_news_url,counter):
    '''Print select details from a html response containing xml
      @param xml_news_url: url to parse
      '''

    context = ssl._create_unverified_context()
    Client = urlopen(xml_news_url, context=context)
    xml_page = Client.read()
    Client.close()

    soup_page = soup(xml_page, "xml")

    news_list = soup_page.findAll("item")
    i = 0  # counter to print n number of news items

    for news in news_list:
        print(f'news title:   {news.title.text}')    # to print title of the news
        print(f'news link:    {news.link.text}')     # to print link of the news
        print(f'news pubDate: {news.pubDate.text}')  # to print published date
        print("+-" * 20, "\n\n")
        
        if i == counter :
          break
        i = i + 1

# you can add google news 'xml' URL here for any country/category
news_url = "https://news.google.com/news/rss/?ned=us&gl=US&hl=en"
sports_url = "https://news.google.com/news/rss/headlines/section/topic/SPORTS.en_in/Sports?ned=in&hl=en-IN&gl=IN"

# now call news function with any of these url or BOTH
news(news_url,10)    
news(sports_url,5)
