import requests
from bs4 import BeautifulSoup

# import csv
import os

# import time
import json


class Phonearena:
    def __init__(self):
        self.phones = []
        self.features = ["Brand", "Model Name", "Model Image"]
        self.temp1 = []
        self.phones_brands = []
        self.url = "https://www.phonearena.com/phones/"  # GSMArena website url
        # Folder name on which files going to save.
        self.new_folder_name = "GSMArenaDataset"
        # It create the absolute path of the GSMArenaDataset folder.
        self.absolute_path = os.getcwd().strip() + "/" + self.new_folder_name

    def crawl_html_page(self, sub_url):

        url = sub_url  # Url for html content parsing.

        # Handing the connection error of the url.
        try:
            page = requests.get(url)
            # It parses the html data from requested url.
            soup = BeautifulSoup(page.text, "html.parser")
            return soup

        except ConnectionError as err:
            print("Please check your network connection and re-run the script.")
            exit()

        except Exception:
            print("Please check your network connection and re-run the script.")
            exit()

    def crawl_phone_urls(self):
        phones_urls = []
        for i in range(1, 238):  # Right now they have 237 page of phone data.
            print(self.url + "page/" + str(i))
            soup = self.crawl_html_page(self.url + "page/" + str(i))
            table = soup.findAll("div", {"class": "stream-item"})
            table_a = [k.find("a") for k in table]
            for a in table_a:
                temp = a["href"]
                phones_urls.append(temp)
        return phones_urls

    def crawl_phones_models_specification(self, li):
        phone_data = {}
        for link in li:
            print(link)
            try:
                soup = self.crawl_html_page(link)
                model = soup.find(class_="page__section page__section_quickSpecs")
                model_name = model.find("header").h1.text
                model_img_html = model.find(class_="head-image")
                model_img = model_img_html.find("img")["data-src"]
                specs_html = model.find(
                    class_="phone__section phone__section_widget_quickSpecs"
                )
                release_date = specs_html.find(class_="calendar")
                release_date = release_date.find(class_="title").p.text
                display = specs_html.find(class_="display")
                display = display.find(class_="title").p.text
                camera = specs_html.find(class_="camera")
                camera = camera.find(class_="title").p.text
                hardware = specs_html.find(class_="hardware")
                hardware = hardware.find(class_="title").p.text
                storage = specs_html.find(class_="storage")
                storage = storage.find(class_="title").p.text
                battery = specs_html.find(class_="battery")
                battery = battery.find(class_="title").p.text
                os = specs_html.find(class_="os")
                os = os.find(class_="title").p.text
                phone_data[model_name] = {
                    "image": model_img,
                    "release_date": release_date,
                    "display": display,
                    "camera": camera,
                    "hardware": hardware,
                    "storage": storage,
                    "battery": battery,
                    "os": os,
                }
                with open(obj.absolute_path + "-PhoneSpecs.json", "w+") as of:
                    json.dump(phone_data, of)
            except Exception as error:
                print(f"Exception happened : {error}")
                continue
        return phone_data


if __name__ == "__main__":
    obj = Phonearena()
    try:
        # Step 1: Scrape links to all the individual phone specs page and save it so that we don't need to run it again.
        phone_urls = obj.crawl_phone_urls()
        with open(obj.absolute_path + "-Phoneurls.json", "w") as of:
            json.dump(phone_urls, of)

        # Step 2: Iterate through all the links from the above execution and run the next command
        with open("obj.absolute_path+'-Phoneurls.json", "r") as inp:
            temp = json.load(inp)
            phone_specs = obj.crawl_phones_models_specification(temp)

    except KeyboardInterrupt:
        print("File has been stopped due to KeyBoard Interruption.")
