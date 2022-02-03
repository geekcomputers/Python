"""
Created on Sat Jul 15 01:41:31 2017

@author: Albert
"""
from __future__ import print_function

import wikipedia as wk
from bs4 import BeautifulSoup


def wiki():
    """
    Search Anything in wikipedia
    """

    word = input("Wikipedia Search : ")
    results = wk.search(word)
    for i in enumerate(results):
        print(i)
    try:
        key = int(input("Enter the number : "))
    except AssertionError:
        key = int(input("Please enter corresponding article number : "))

    page = wk.page(results[key])
    url = page.url
    # originalTitle=page.original_title
    pageId = page.pageid
    # references=page.references
    title = page.title
    # soup=BeautifulSoup(page.content,'lxml')
    pageLength = input("""Wiki Page Type : 1.Full 2.Summary : """)
    if pageLength == 1:
        soup = fullPage(page)
        print(soup)
    else:
        print(title)
        print("Page Id = ", pageId)
        print(page.summary)
        print("Page Link = ", url)
    # print "References : ",references

    pass


def fullPage(page):
    soup = BeautifulSoup(page.content, "lxml")
    return soup


def randomWiki():
    """
    This function gives you a list of n number of random articles
    Choose any article.
    """
    number = input("No: of Random Pages : ")
    lst = wk.random(number)
    for i in enumerate(lst):
        print(i)
    try:
        key = input("Enter the number : ")
        assert key >= 0 and key < number
    except AssertionError:
        key = input("Please enter corresponding article number : ")

    page = wk.page(lst[key])
    url = page.url
    # originalTitle=page.original_title
    pageId = page.pageid
    # references=page.references
    title = page.title
    # soup=BeautifulSoup(page.content,'lxml')
    pageLength = input("""Wiki Page Type : 1.Full 2.Summary : """)
    if pageLength == 1:
        soup = fullPage(page)
        print(soup)
    else:
        print(title)
        print("Page Id = ", pageId)
        print(page.summary)
        print("Page Link = ", url)
    # print "References : ",references

    pass


# if __name__=="__main__":
#    wiki()
