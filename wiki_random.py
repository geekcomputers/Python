"""Author Anurag Kumar(mailto:anuragkumarak95@gmail.com)
Module for Fetching Random Wiki Pages and asking user for opening one of them

Python:
  - 3.5

Requirements:
  - requests
  - json
  - webbrowser

Usage:
  - $python3 wiki_random.py

enter index of article you would like to see, or 'r' for retry and 'n' for exit.
"""
import requests
import webbrowser

page_count = 10
url = (
    "https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit="
    + str(page_count)
    + "&format=json"
)


def load():
    response = requests.get(url)
    if response.ok:
        jsonData = response.json()["query"]["random"]
        print("10 Random generted WIKI pages...")
        for idx, j in enumerate(jsonData):
            print(str(idx) + ": ", j["title"])
        i = input(
            "Which page you want to see, enter index..[r: for retry,n: exit]?"
        ).lower()
        if i == "r":
            print("Loading randoms again...")
        elif i == "n":
            print("Auf Wiedersehen!")
            return
        else:
            try:
                jsonData[int(i)]["id"]
            except Exception:
                raise Exception("Wrong Input...")
            print("taking you to the browser...")
            webbrowser.get().open(
                "https://en.wikipedia.org/wiki?curid=" + str(jsonData[int(i)]["id"])
            )
        load()
    else:
        response.raise_for_status()


if __name__ == "__main__":
    load()
