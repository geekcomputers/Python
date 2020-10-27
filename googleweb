from fuzzywuzzy import fuzz
import bs4, requests
import numpy as np
import pandas as pd
import os
requests.packages.urllib3.disable_warnings()
FinalResult=[]
def SearchResults():
    lis = []
    f = open("Input", "r")
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    StabUrl = "https://www.google.com/search?rlz=1C1CHBD_enIN872IN872&sxsrf=ALeKk03OHYAnSxX60oUwmblKn36Hyi8MhA%3A1600278715451&ei=u1BiX9ibG7qU4-EP_qGPgA8&q="
    midUrl = "&oq="
    EndUrl = "&gs_lcp=CgZwc3ktYWIQAzoECAAQR1C11AxYtdQMYJXcDGgAcAF4AIABpQKIAaUCkgEDMi0xmAEAoAECoAEBqgEHZ3dzLXdpesgBCMABAQ&sclient=psy-ab&ved=0ahUKEwiY5YDjnu7rAhU6yjgGHf7QA_AQ4dUDCA0&uact=5"
    for i in f:
        singleLink=[]
        singleRatio=[]
        singleWrite=[]
        singleWrite.append(i.strip("\n"))
        checkString=i.replace("+","")
        searchString=i.replace("+","%2B")
        searchString=searchString.replace(" ","+")
        searchString=StabUrl+searchString+midUrl+searchString+EndUrl
        r = requests.get(searchString, headers=header)
        soup = bs4.BeautifulSoup(r.text, features="html.parser")
        elements = soup.select(".r a")
        for g in elements:
            lis.append(g.get("href"))
        for k in lis:
            sentence=""
            if (k[0] != "#") and k[0] != "/":
                checker = k[8:16]
                if (checker != "webcache"):
                    rr = requests.get(k, headers=header, verify=False)
                    soupInside = bs4.BeautifulSoup(rr.text, features="html.parser")
                    elementInside=soupInside.select("body")
                    for line in elementInside:
                        sentence=sentence+line.text
                    ratio=fuzz.token_set_ratio(sentence,checkString)
                    if(ratio>80):
                        singleLink.append(k)
                        singleRatio.append(ratio)
        if(len(singleLink)>=4):
            singleLink=np.array(singleLink)
            singleRatio=np.array(singleRatio)
            inds=singleRatio.argsort()
            sortedLink=singleLink[inds]
            sortedFinalList=list(sortedLink[::-1])
            sortedFinalList=sortedFinalList[:4]
            FinalResult.append(singleWrite+sortedFinalList)
        elif(len(singleLink)<4) and len(singleLink)>0:
            singleLink = np.array(singleLink)
            singleRatio = np.array(singleRatio)
            inds = singleRatio.argsort()
            sortedLink = singleLink[inds]
            sortedFinalList = list(sortedLink[::-1])
            sortedFinalList=sortedFinalList+(4-len(sortedFinalList))*[[" "]]
            FinalResult.append(singleWrite + sortedFinalList)
        else:
            sortedFinalList=[[" "]]*4
            FinalResult.append(singleWrite+sortedFinalList)


SearchResults()
FinalResult=np.array(FinalResult)
FinalResult=pd.DataFrame(FinalResult)
FinalResult.columns=["Input","Link A","Link B","Link C","Link D"]
FinalResult.replace(" ",np.nan)
FinalResult.to_csv("Susma.csv",index=False)
print(FinalResult)
