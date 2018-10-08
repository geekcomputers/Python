#!/usr/bin/python3
import urllib2

def checkInternetConnectivity():
    try:
        urllib2.urlopen("http://google.com", timeout=2)
        print("Working connection")

    except urllib2.URLError as E:
        print("Connection error:%s"%E.reason)

checkInternetConnectivity()
