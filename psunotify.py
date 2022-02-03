from __future__ import print_function

import re

import mechanize
import urllib2

br = mechanize.Browser()
br.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36",
    )
]
br.set_handle_robots(False)
# For page exploration
page = input("Enter Page No:")
# print type(page)
p = urllib2.Request(
    "https://www.google.co.in/search?q=gate+psu+2017+ext:pdf&start=" + page
)
ht = br.open(p)
text = '<cite\sclass="_Rm">(.+?)</cite>'
patt = re.compile(text)
h = ht.read()
urls = re.findall(patt, h)
int = 0
while int < len(urls):
    urls[int] = urls[int].replace("<b>", "")
    urls[int] = urls[int].replace("</b>", "")
    int = int + 1

print(urls)

for url in urls:
    try:
        temp = url.split("/")
        q = temp[len(temp) - 1]
        if "http" in url:
            r = urllib2.urlopen(url)
        else:
            r = urllib2.urlopen("http://" + url)
        file = open("psu2" + q + ".pdf", "wb")
        file.write(r.read())
        file.close()

        print("Done")
    except urllib2.URLError as e:
        print(
            "Sorry there exists a problem with this URL Please Download this Manually "
            + str(url)
        )
