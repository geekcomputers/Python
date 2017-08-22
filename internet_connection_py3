import urllib2
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
print "Testing Internet Connection"
print
try:
    urllib2.urlopen("http://google.com", timeout=2)#Tests if connection is up and running
    print "Internet is working fine!"
    print
    question = raw_input("Do you want to open a website? (Y/N): ")
    if question == 'Y':
    	print
    	search = raw_input("Input website to open (http://website.com) : ")
    else:
    	os._exit(0)

except urllib2.URLError:
    print ("No internet connection!")#Output if no connection

browser = webdriver.Firefox()
browser.get(search)
os.system('cls')#os.system('clear') if Linux
print "[+] Website "+search + " opened!"
browser.close()
