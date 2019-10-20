from __future__ import print_function

import os
import urllib.request

from selenium import webdriver

print("Testing Internet Connection")
print()
try:
    urllib.request.urlopen("http://google.com", timeout=2)  # Tests if connection is up and running
    print("Internet is working fine!")
    print()
    question = input("Do you want to open a website? (Y/N): ")
    if question == 'Y':
        print()
        search = input("Input website to open (http://website.com) : ")
    else:
        os._exit(0)

except urllib.error.URLError:
    print("No internet connection!")  # Output if no connection

browser = webdriver.Firefox()
browser.get(search)
os.system('cls')  # os.system('clear') if Linux
print("[+] Website " + search + " opened!")
browser.close()
