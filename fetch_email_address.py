
__author__ = 'tusharsappal'


## This script fetches the email addresses from the string provided , the script only fetches the email addresses of the format alphanumeric@alphabets

import re
def fetch_email_address(str):
    fetcher=re.findall('[a-zA-Z0-9]\S+@\S+[a-zA-Z]]',str)
    if len(fetcher)>0:
        print fetcher



## Replace the method argument with the string to be parsed, or you can modify the script to read the data from the text file


fetch_email_address("Replace this argument with the string to be parsed")





