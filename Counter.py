__author__ = 'tusharsappal'
## This script uses the dictionary utility of the python and counts the number of the characters in the string and their frequency
def Counter(str):
    d=dict()
    for c in str:
        if c not in d:
            d[c]=1
        else :
            d[c]=d[c]+1

    return  d


## Just replace the method argument with the string for which you want to check 


temp=Counter("Replace with the string for which you want to check ")
print temp