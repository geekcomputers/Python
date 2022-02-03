import requests
from bs4 import BeautifulSoup
import re

re_text = r"\:|\.|\!|(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b|(.twitter.com\/)\w*|\&"
re_text_1 = r"(pictwittercom)\/\w*"


def tweeter_scrapper():
    list_of_dirty_tweets = []
    clear_list_of_tweets = []
    base_tweeter_url = "https://twitter.com/{}"

    tweeter_id = input()

    response = requests.get(base_tweeter_url.format(tweeter_id))
    soup = BeautifulSoup(response.content, "lxml")
    all_tweets = soup.find_all("div", {"class": "tweet"})

    for tweet in all_tweets:
        content = tweet.find("div", {"class": "content"})
        message = (
            content.find("div", {"class": "js-tweet-text-container"})
            .text.replace("\n", " ")
            .strip()
        )
        list_of_dirty_tweets.append(message)
    for dirty_tweet in list_of_dirty_tweets:
        dirty_tweet = re.sub(re_text, "", dirty_tweet, flags=re.MULTILINE)
        dirty_tweet = re.sub(re_text_1, "", dirty_tweet, flags=re.MULTILINE)
        dirty_tweet = dirty_tweet.replace(u"\xa0…", u"")
        dirty_tweet = dirty_tweet.replace(u"\xa0", u"")
        dirty_tweet = dirty_tweet.replace(u"\u200c", u"")
        clear_list_of_tweets.append(dirty_tweet)
    print(clear_list_of_tweets)


if __name__ == "__main__":
    tweeter_scrapper()
