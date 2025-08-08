# news_oversimplifier.py
# Python command-line tool that fetches recent news articles based on a search query using NewsAPI and summarizes the article content using extractive summarization. You can also save the summaries to a text file.

# (requires API key in .env file)

import requests
import os
import sys
from dotenv import load_dotenv
from summa.summarizer import summarize


def main():

    # loads .env variables
    load_dotenv()
    API_KEY = os.getenv("NEWS_API_KEY")

    # check validity of command-line arguments
    try:
        if len(sys.argv) == 2:
            news_query = sys.argv[1]
        else:
            raise IndexError()
    except IndexError:
        sys.exit('Please provide correct number of command-line arguments')

    try:
        # get number of articles from user
        while True:
            try:
                num_articles = int(input('Enter number of articles: '))
                break
            except ValueError:
                continue

        # fetch news articles based on user's query
        articles = fetch_news(API_KEY, query=news_query, max_articles=num_articles)

        # output printing title, summary and no. of words in the summary
        for i, article in enumerate(articles):
            capitalized_title = capitalize_title(article["title"])
            print(f"\n{i+1}. {capitalized_title}")

            content = article.get("content") or article.get("description") or ""
            if not content.strip():
                print("No content to oversimplify.")
                continue

            summary = summarize_text(content) # returns summary
            count = word_count(summary) # returns word count
            print(f"\nOVERSIMPLIFIED:\n{summary}\n{count} words\n")

            # ask user whether they want to save the output in a txt file
            while True:
                saving_status = input(
                    "Would you like to save this in a text file? (y/n): ").strip().lower()
                if saving_status == "y":
                    save_summary(article["title"], summary)
                    break
                elif saving_status == "n":
                    break
                else:
                    print('Try again\n')
                    continue

    except Exception as e:
        print("ERROR:", e)


def word_count(text):  # pytest in test file
    """
    Returns the number of words in the given text.

    args:
        text (str): Input string to count words from.

    returns:
        int: Number of words in the string.
    """
    return len(text.split())


def summarize_text(text, ratio=0.6):  # pytest in test file
    """
    Summarizes the given text using the summa library.

    args:
        text (str): The input text to summarize.
        ratio (float): Ratio of the original text to retain in the summary.

    returns:
        str: The summarized text, or a fallback message if intro is present or summary is empty.
    """
    summary = summarize(text, ratio=ratio)
    if summary.lower().startswith("hello, and welcome to decoder!"):
        return "No description available for this headline"
    else:
        return summary.strip() if summary else text


def capitalize_title(title):  # pytest in test file
    """
    Capitalizes all letters in a given article title.

    args:
        title (str): The title to format.

    returns:
        str: Title in uppercase with surrounding spaces removed.
    """
    return title.upper().strip()


def fetch_news(api_key, query, max_articles=5):  # no pytest
    """
    Fetches a list of news articles from NewsAPI based on a query string.

    args:
        api_key (str): NewsAPI key loaded from environment.
        query (str): The keyword to search for in news articles.
        max_articles (int): Maximum number of articles to fetch.

    returns:
        list: List of dictionaries, each representing a news article.

    raises:
        Exception: If the API response status is not 'ok'.
    """
    url = (
        f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}&pageSize={max_articles}"
    )
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok":
        raise Exception("Failed to fetch news:", data.get("message"))
    return data["articles"]


def save_summary(title, summary, path="summaries.txt"):  # no pytest
    """
    Appends a formatted summary to a file along with its title.

    args:
        title (str): Title of the article.
        summary (str): Summarized text to save.
        path (str): File path to save the summary, i.e. 'summaries.txt'
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{title}\n{summary}\n{'='*60}\n")


if __name__ == "__main__":
    main()