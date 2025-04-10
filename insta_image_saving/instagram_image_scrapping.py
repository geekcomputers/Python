# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
from io import BytesIO
from time import sleep

import pandas as pd
import requests
import scrapy
from PIL import Image
from scrapy import Spider
from scrapy.selector import Selector
from selenium import webdriver

# +
imageID = []
sl_no = []
imageLikes = []
i = 0
instaccountlink = "https://instagram.com/audi"
instaaccountname = "Audi"
driver = webdriver.Chrome("driver/driver")
driver.get(instaccountlink)
unique_urls = []
while i < 300:
    i = i + 1
    sel = Selector(text=driver.page_source)

    url = sel.xpath('//div[@class="v1Nh3 kIKUG  _bz0w"]/a/@href').extract()
    for u in url:
        if u not in unique_urls:
            unique_urls.append(u)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sel = Selector(text=driver.page_source)
    url = sel.xpath('//div[@class="v1Nh3 kIKUG  _bz0w"]/a/@href').extract()
    sleep(1)
    for u in url:
        if u not in unique_urls:
            unique_urls.append(u)

driver.quit()
print(len(unique_urls))
# -

file = open("output/audi_instagram_11_07_2019.csv", "a")
for u in unique_urls:
    file.write(u)
    file.write("\n")
file.close()
print("file saved successfully")

# +
# saving the images to specified directory
driver = webdriver.Chrome("driver/driver")

image_urls = []
count = 0
max_no_of_iteration = 250
for u in unique_urls:
    try:
        driver.get("http://instagram.com" + u)
        sel = Selector(text=driver.page_source)

        src = sel.xpath("//div/img/@src").extract()[0]
        #             print(src)
        r = requests.get(src)

        image = Image.open(BytesIO(r.content))
        #         path = "C:/Users/carbon/Desktop/output/"+instaAccountName+str(count)+"." + image.format
        path = "output/" + instaaccountname + str(count) + "." + image.format
        #             print(image.size, image.format, image.mode)
        q1 = ""
        q2 = ""
        try:
            image.save(path, image.format)
            q1 = instaaccountname + str(count)
            q2 = sel.xpath("//span/span/text()").extract_first()
        #             print(q1)
        #             print(q2)

        except OSError:
            q1 = ""
            q2 = ""
        imageID.insert(len(imageID), q1)
        imageLikes.insert(len(imageLikes), q2)
        sl_no.insert(len(sl_no), str(count))
        count = count + 1
        if count > max_no_of_iteration:
            driver.quit()
            df = pd.DataFrame(
                {"ImageID": imageID, "Sl_no": sl_no, "ImageLikes": imageLikes}
            )
            fileName = instaaccountname + ".csv"
            df.to_csv(fileName, index=False)
            break

    except:
        pass

try:
    driver.quit()
except:
    pass
# -


