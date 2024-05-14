import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import html5lib

# * Using html5lib as the parser is good
# * It is the most lenient parser and works as

animals_A_to_Z_URL = "https://animalcorner.org/animal-sitemap/#"

results = requests.get(animals_A_to_Z_URL)
# ? results and results.text ? what are these?

# soup = BeautifulSoup(results.text, "html.parser")
# * will use html5lib as the parser
soup = BeautifulSoup(results.text, "html5lib")

# print(soup.prettify())

# To store animal names
animal_name = []

# To store the titles of animals
animal_title = []

# alphabet_head = soup.find_all("div", class_="wp-block-heading")
# alphabet_head = soup.find_all("div", class_="side-container")
# * .text all it's immediate text and children
# * .string only the immediate text
# print(soup.find_all("h2", class_="wp-block-heading"))
# az_title = soup.find_all("h2", class_="wp-block-heading")
az_names = soup.find_all(
    "div", class_="wp-block-column is-layout-flow wp-block-column-is-layout-flow"
)
# az_title = soup
# for title in az_title:
#     # print(title.text)
# print(title.string)
# print(title.find(class_="wp-block-testing"))

for name_div in az_names:
    a_names = name_div.find_all("br")

    for elements in a_names:
        # print(elements.text)
        # print(elements, end="\n")
        next_sibling = elements.next_sibling
        # Check if the next sibling exists and if it's not a <br> element
        while next_sibling and next_sibling.name == "br":
            next_sibling = next_sibling.next_sibling

            
        # Print the text content of the next sibling element
        if next_sibling:
            print(next_sibling.text.strip())

    # print(name.text)

# print(soup.h2.string)

# for container in alphabet_head:
# print(container.text, end="\n")
# titles = container.div.div.find("h2", class_="wp-block-heading")
# title = container.find("h2", class_="wp-block-heading")
# title = container.h3.text
# print(title.text, end="\n")

# print(container.find_all("h2", class_ = "wp-block-heading"))


# print(soup.get_text(), end="\p")

# Want to write it to a file and sort and analyse it
