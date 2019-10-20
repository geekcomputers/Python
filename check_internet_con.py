from sys import argv
try:
    # For Python 3.0 and later
    from urllib.error import URLError
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import URLError, urlopen


def checkInternetConnectivity():
    try:
        url = argv[1]
        if 'https://' or 'http://' not in url:
            url = 'https://' + url
    except:
        url = 'https://google.com'
    try:
         urlopen(url, timeout=2)
         print("Connection to \""+ url + "\" is working")
        
    except URLError as E:
        print("Connection error:%s" % E.reason)


checkInternetConnectivity()
