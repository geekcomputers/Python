import urllib2

try:
    urllib2.urlopen("8.8.8.8", timeout=2)
    print ("working connection")

except urllib2.URLError:
    print ("No internet connection")
