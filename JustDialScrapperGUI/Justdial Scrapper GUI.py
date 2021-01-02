from tkinter import Tk, Entry, Label, Button, HORIZONTAL
from tkinter.ttk import Progressbar
from bs4 import BeautifulSoup

import urllib.request
import threading
import csv


class ScrapperLogic:
    def __init__(self, query, location, file_name, progressbar, label_progress):
        self.query = query
        self.location = location
        self.file_name = file_name
        self.progressbar = progressbar
        self.label_progress = label_progress

    @staticmethod
    def inner_html(element):
        return element.decode_contents(formatter="html")

    @staticmethod
    def get_name(body):
        return body.find('span', {'class': 'jcn'}).a.string

    @staticmethod
    def which_digit(html):
        mapping_dict = {'icon-ji': 9,
                       'icon-dc': '+',
                       'icon-fe': '(',
                       'icon-hg': ')',
                       'icon-ba': '-',
                       'icon-lk': 8,
                       'icon-nm': 7,
                       'icon-po': 6,
                       'icon-rq': 5,
                       'icon-ts': 4,
                       'icon-vu': 3,
                       'icon-wx': 2,
                       'icon-yz': 1,
                       'icon-acb': 0,
                       }
        return mapping_dict.get(html, '')

    def get_phone_number(self, body):
        i = 0
        phone_no = "No Number!"
        try:
            for item in body.find('p', {'class': 'contact-info'}):
                i += 1
                if i == 2:
                    phone_no = ''
                    try:
                        for element in item.find_all(class_=True):
                            classes = []
                            classes.extend(element["class"])
                            phone_no += str((self.which_digit(classes[1])))
                    except:
                        pass
        except:
            pass
        body = body['data-href']
        soup = BeautifulSoup(body, 'html.parser')
        for a in soup.find_all('a', {"id": "whatsapptriggeer"}):
            # print (a)
            phone_no = str(a['href'][-10:])

        return phone_no

    @staticmethod
    def get_rating(body):
        rating = 0.0
        text = body.find('span', {'class': 'star_m'})
        if text is not None:
            for item in text:
                rating += float(item['class'][0][1:]) / 10

        return rating

    @staticmethod
    def get_rating_count(body):
        text = body.find('span', {'class': 'rt_count'}).string

        # Get only digits
        rating_count = ''.join(i for i in text if i.isdigit())
        return rating_count      
    
    @staticmethod
    def get_address(body):
        return body.find('span', {'class': 'mrehover'}).text.strip()

    @staticmethod
    def get_location(body):
        text = body.find('a', {'class': 'rsmap'})
        if not text:
            return
        text_list = text['onclick'].split(",")

        latitude = text_list[3].strip().replace("'", "")
        longitude = text_list[4].strip().replace("'", "")

        return latitude + ", " + longitude

    def start_scrapping_logic(self):
        page_number = 1
        service_count = 1

        total_url = "https://www.justdial.com/{0}/{1}".format(self.location, self.query)

        fields = ['Name', 'Phone', 'Rating', 'Rating Count', 'Address', 'Location']
        out_file = open('{0}.csv'.format(self.file_name), 'w')
        csvwriter = csv.DictWriter(out_file, delimiter=',', fieldnames=fields)
        csvwriter.writerow({
            'Name': 'Name', #Shows the name
            'Phone': 'Phone',#shows the phone 
            'Rating': 'Rating',#shows the ratings
            'Rating Count': 'Rating Count',#Shows the stars for ex: 4 stars
            'Address': 'Address',#Shows the address of the place
            'Location': 'Location'#shows the location
        })

        progress_value = 0
        while True:
            # Check if reached end of result
            if page_number > 50:
                progress_value = 100
                self.progressbar['value'] = progress_value
                break

            if progress_value != 0:
                progress_value += 1
                self.label_progress['text'] = "{0}{1}".format(progress_value, '%')
                self.progressbar['value'] = progress_value

            url = total_url + "/page-%s" % page_number
            print("{0} {1}, {2}".format("Scrapping page number: ", page_number, url))
            req = urllib.request.Request(url, headers={'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"})
            page = urllib.request.urlopen(req)

            soup = BeautifulSoup(page.read(), "html.parser")
            services = soup.find_all('li', {'class': 'cntanr'})

            # Iterate through the 10 results in the page

            progress_value += 1
            self.label_progress['text'] = "{0}{1}".format(progress_value, '%')
            self.progressbar['value'] = progress_value

            for service_html in services:
                try:
                    # Parse HTML to fetch data
                    dict_service = {}
                    name = self.get_name(service_html)
                    print(name)
                    phone = self.get_phone_number(service_html)
                    rating = self.get_rating(service_html)
                    count = self.get_rating_count(service_html)
                    address = self.get_address(service_html)
                    location = self.get_location(service_html)
                    if name is not None:
                        dict_service['Name'] = name
                    if phone is not None:
                        print('getting phone number')
                        dict_service['Phone'] = phone
                    if rating is not None:
                        dict_service['Rating'] = rating
                    if count is not None:
                        dict_service['Rating Count'] = count
                    if address is not None:
                        dict_service['Address'] = address
                    if location is not None:
                        dict_service['Address'] = location

                    # Write row to CSV
                    csvwriter.writerow(dict_service)

                    print("#" + str(service_count) + " ", dict_service)
                    service_count += 1
                except AttributeError:
                    print("AttributeError Occurred 101")

            page_number += 1

        out_file.close()


class JDScrapperGUI:
    def __init__(self, master):
        self.master = master

        self.label_query = Label
        self.entry_query = Entry

        self.label_location = Label
        self.entry_location = Entry

        self.label_file_name = Label
        self.entry_file_name = Entry

        self.label_progress = Label
        self.button_start = Button

        # Progress bar widget
        self.progress = Progressbar

    def start_scrapping(self):
        query = self.entry_query.get()
        location = self.entry_location.get()
        file_name = self.entry_file_name.get()
        scrapper = ScrapperLogic(query, location, file_name, self.progress, self.label_progress)
        t1 = threading.Thread(target=scrapper.start_scrapping_logic, args=[])
        t1.start()

    def start(self):
        self.label_query = Label(self.master, text='Query')
        self.label_query.grid(row=0, column=0)

        self.entry_query = Entry(self.master, width=23)
        self.entry_query.grid(row=0, column=1)

        self.label_location = Label(self.master, text='Location')
        self.label_location.grid(row=1, column=0)

        self.entry_location = Entry(self.master, width=23)
        self.entry_location.grid(row=1, column=1)

        self.label_file_name = Label(self.master, text='File Name')
        self.label_file_name.grid(row=2, column=0)

        self.entry_file_name = Entry(self.master, width=23)
        self.entry_file_name.grid(row=2, column=1)

        self.label_progress = Label(self.master, text='0%')
        self.label_progress.grid(row=3, column=0)

        self.button_start = Button(self.master, text="Start", command=self.start_scrapping)
        self.button_start.grid(row=3, column=1)

        self.progress = Progressbar(self.master, orient=HORIZONTAL, length=350, mode='determinate')
        self.progress.grid(row=4, columnspan=2)
      #Above is the progress bar

if __name__ == '__main__':
    root = Tk()
    root.geometry('350x130+600+100')
    root.title("Just Dial Scrapper - Cool")
    JDScrapperGUI(root).start()
    root.mainloop()
