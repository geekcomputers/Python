import requests, json, webbrowser

url = 'https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=10&format=json'

def load():
    response = requests.get(url)
    if response.ok:
        jsonData = response.json()['query']['random']
        print("10 Random generted WIKI pages...")
        for idx,j in enumerate(jsonData):
            print(str(idx)+": ",j['title'])
        i = input("Which page you want to see, enter index..[r: for retry,n: exit]?").lower()
        if i == 'r':
            print('Loading randoms again...')
        elif i=='n':
            print('Auf Wiedersehen!')
            return
        else:
            try: int(i)
            except Exception: raise Exception("Wrong Input...")
            print('taking you to the browser...')
            webbrowser.get().open('https://en.wikipedia.org/wiki?curid='+str(jsonData[int(i)]['id']))
        load()
    else:
        response.raise_for_status()

if __name__=='__main__':
    load()
