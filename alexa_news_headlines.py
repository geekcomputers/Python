import json
import time

import requests
import unidecode
from flask import Flask
from flask_ask import Ask, question, statement

app = Flask(__name__)
ask = Ask(app, "/reddit_reader")


def get_headlines():
    user_pass_dict = {"user": "USERNAME", "passwd": "PASSWORD", "api_type": "json"}
    sess = requests.Session()
    sess.headers.update({"User-Agent": "I am testing Alexa: nobi"})
    sess.post("https://www.reddit.com/api/login/", data=user_pass_dict)
    time.sleep(1)
    url = "https://reddit.com/r/worldnews/.json?limit=10"
    html = sess.get(url)
    data = json.loads(html.content.decode("utf-8"))
    titles = [
        unidecode.unidecode(listing["data"]["title"])
        for listing in data["data"]["children"]
    ]
    titles = "... ".join([i for i in titles])
    return titles


@app.route("/")
def homepage():
    return "hi there!"


@ask.launch
def start_skill():
    welcome_message = "Hello there, would you like to hear the news?"
    return question(welcome_message)


@ask.intent("YesIntent")
def share_headlines():
    headlines = get_headlines()
    headline_msg = "The current world news headlines are {}".format(headlines)
    return statement(headline_msg)


@ask.intent("NooIntent")
def no_intent():
    bye_text = "I am not sure why you then turned me on. Anyways, bye for now!"
    return statement(bye_text)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
