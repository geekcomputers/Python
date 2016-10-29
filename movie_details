import mechanize
from bs4 import BeautifulSoup 
import urllib2
# Create a Browser
b = mechanize.Browser()

# Disable loading robots.txt
b.set_handle_robots(False)

b.addheaders = [('User-agent',
                 'Mozilla/4.0 (compatible; MSIE 5.0; Windows 98;)')]
nm=raw_input("enter title ")
# Navigate
b.open('http://www.imdb.com/search/title')

# Choose a form
b.select_form(nr=1)


b['title'] = nm

b.find_control(type="checkbox",nr=0).get("feature").selected = True


# Submit
fd = b.submit()

soup = BeautifulSoup(fd.read(),'html5lib')

#data= soup.find_all('td',class_="title")
#for div in data:
#  links= div.find_all('a')
 #  for a in links:
  #      print a['href'];


for div in soup.findAll('td', {'class': 'title'},limit=1):
    a = div.findAll('a')[0]
    print a.text.strip(), '=>', a.attrs['href']
    hht='http://www.imdb.com'+a.attrs['href']
    print(hht)
    page=urllib2.urlopen(hht)
    soup2 = BeautifulSoup(page.read(),'html.parser')
    print("title of the movie: ")
    print(soup2.find(itemprop="name").get_text())
    print("timerun: ")
    print(soup2.find(itemprop="duration").get_text())
    print("genre: ")
    print(soup2.find(itemprop="genre").get_text())
    print("current IMDB rating:")
    print(soup2.find(itemprop="ratingValue").get_text())
    print("summary:")
    print(soup2.find(itemprop="description").get_text())
