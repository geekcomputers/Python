#!/usr/bin/python3

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

def checkInternetConnectivity():
    try:
        urllib2.urlopen("http://google.com", timeout=2)
        print("Working connection")

    except urllib2.URLError as E:
        print("Connection error:%s" % E.reason)


checkInternetConnectivity()
