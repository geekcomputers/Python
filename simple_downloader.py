__author__ = 'tusharsappal'

import urllib
def simple_downloader(url,path_to_store_content):
    img = urllib.urlopen(url)
    fhand=open(path_to_store_content,'w')
    size=0
    while True:
        info=img.read(10000)
        if len(info)<1:
            break
        else :
            size=size+len(info)
        fhand.write(info)


    fhand.close()



##replace the first function parameter with the path of the the url  from which the content is to be fetched
## And the second parameter with the path on your local system to store the content fetched

simple_downloader("Replace with the URL","Replace with the path on your local system where we want to store the content")


