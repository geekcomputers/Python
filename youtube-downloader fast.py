'''
Author: Anshu Saini
GitHub: https://github.com/anshu189
mail: anshusaini189381@gmail.com
Requirements: pytube (pip install pytube)
'''

import pytube as py

url = input("Enter your youtube video url: ")

py.YouTube(url).streams.get_highest_resolution().download(
    'C:/Users/user-name/Desktop')  # (PATH) Where you want to save your downloaded video
