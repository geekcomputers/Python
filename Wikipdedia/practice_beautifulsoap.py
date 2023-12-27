from bs4 import BeautifulSoup
import requests

language_symbols = {}


def lang():
    try:
        response = requests.get("https://www.wikipedia.org/")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for option in soup.find_all('option'):
            language = option.text
            symbol = option['lang']
            language_symbols[language] = symbol

        return list(language_symbols.keys())

    except requests.exceptions.RequestException as e:
        print("Error fetching language data:", e)
        return []


def data(selected_topic, selected_language):
    symbol = language_symbols.get(selected_language)

    try:
        url = f"https://{symbol}.wikipedia.org/wiki/{selected_topic}"
        data_response = requests.get(url)
        data_response.raise_for_status()
        data_soup = BeautifulSoup(data_response.content, 'html.parser')

        main_content = data_soup.find('div', {'id': 'mw-content-text'})
        filtered_content = ""

        if main_content:
            for element in main_content.descendants:
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    filtered_content += "\n" + element.get_text(strip=True).upper() + "\n"

                elif element.name == 'p':
                    filtered_content += element.get_text(strip=True) + "\n"

        return filtered_content

    except requests.exceptions.RequestException as e:
        print("Error fetching Wikipedia content:", e)
        return "Error fetching data."


def get_image_urls(query):
    try:
        search_url = f"https://www.google.com/search?q={query}&tbm=isch"
        image_response = requests.get(search_url)
        image_response.raise_for_status()
        image_soup = BeautifulSoup(image_response.content, 'html.parser')

        image_urls = []
        for img in image_soup.find_all('img'):
            image_url = img.get('src')
            if image_url and image_url.startswith("http"):
                image_urls.append(image_url)

        return image_urls[0]

    except requests.exceptions.RequestException as e:
        print("Error fetching image URLs:", e)
        return None
