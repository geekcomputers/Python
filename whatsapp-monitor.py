#! /usr/bin/python3

'''
Author- Tony Stark 

download https://github.com/mozilla/geckodriver/releases

set path paste binary file /usr/local/bin 

install requirements: python -m pip install selenium

'''

from selenium import webdriver
import os
import time
driver = webdriver.Firefox()
driver.get("http://web.whatsapp.com")
name=input("Please Enter Name for search online status: ")

while True:

    try:
        chat=driver.find_element_by_xpath("/html/body/div[1]/div/div/div[3]/div/header/div[2]/div/span/div[2]/div")
        chat.click()
        time.sleep(2)
        search=driver.find_element_by_xpath("/html/body/div[1]/div/div/div[2]/div[1]/span/div/span/div/div[1]/div/label/input")
        search.click()
        time.sleep(2)
        search.send_keys(name)
        time.sleep(2)
        open=driver.find_element_by_xpath("/html/body/div[1]/div/div/div[2]/div[1]/span/div/span/div/div[2]/div[1]/div/div/div[2]/div/div")
        open.click()
        time.sleep(2)


        while True:
            try:
                status = driver.find_element_by_class_name("_315-i").text
                name = driver.find_element_by_class_name("_19vo_").text
                print("{0} is {1}".format(name,status))
                time.sleep(30)
            except:
            	name = driver.find_element_by_class_name("_19vo_").text
            	print("{0} is {1}".format(name,"offline"))
            	time.sleep(30)


    except:
            pass


            

