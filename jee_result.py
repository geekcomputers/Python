import datetime

import mechanize
from bs4 import BeautifulSoup

# Create a Browser
b = mechanize.Browser()

# Disable loading robots.txt
b.set_handle_robots(False)

b.addheaders = [('User-agent',
                 'Mozilla/4.0 (compatible; MSIE 5.0; Windows 98;)')]

# Navigate
b.open('http://cbseresults.nic.in/jee/jee_2015.htm')

# Choose a form
b.select_form(nr=0)

# Fill it out
b['regno'] = '37000304'

currentdate = datetime.date(1997, 3, 10)
enddate = datetime.date(1998, 4, 1)
while currentdate <= enddate:
    ct = 0
    # print currentdate
    yyyymmdd = currentdate.strftime("%Y/%m/%d")
    ddmmyyyy = yyyymmdd[8:] + "/" + yyyymmdd[5:7] + "/" + yyyymmdd[:4]
    print(ddmmyyyy)
    b.open('http://cbseresults.nic.in/jee/jee_2015.htm')
    b.select_form(nr=0)
    b['regno'] = '37000304'
    b['dob'] = ddmmyyyy

    fd = b.submit()
    # print(fd.read())
    soup = BeautifulSoup(fd.read(), 'html.parser')

    for writ in soup.find_all('table'):
        ct = ct + 1;
    # print (ct)
    if ct == 6:
        print("---fail---")
    else:
        print("--true--")
        break;
    currentdate += datetime.timedelta(days=1)
    # print fd.read()
