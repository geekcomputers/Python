import ssl
from urllib.request import urlopen

from bs4 import BeautifulSoup as soup

# --- Helper Functions for Error handling---


def fetch_xml(url):
    """Fetch XML content safely from a URL."""
    try:
        context = ssl._create_unverified_context()
        with urlopen(url, context=context) as client:
            return client.read()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None


def get_text_or_default(tag, default="N/A"):
    """Safely extract text from a tag."""
    return tag.text if tag else default


# --- news printing function---


def news(xml_news_url, counter):
    """Print select details from a html response containing xml
    @param xml_news_url: url to parse
    """

    xml_page = fetch_xml(xml_news_url)
    if xml_page is None:
        return

    soup_page = soup(xml_page, "xml")

    news_list = soup_page.findAll("item")

    i = 0  # counter to print n number of news items

    for i, item in enumerate(news_list):
        if i >= counter:
            break

        title = get_text_or_default(item.title)
        link = get_text_or_default(item.link)
        pub_date = get_text_or_default(item.pubDate)

        print(f"news title:   {title}")
        print(f"news link:    {link}")
        print(f"news pubDate: {pub_date}")
        print("+-" * 20, "\n\n")

        i = i + 1


if __name__ == "__main__":
    # you can add google news 'xml' URL here for any country/category
    news_url = "https://news.google.com/news/rss/?ned=us&gl=US&hl=en"
    sports_url = "https://news.google.com/news/rss/headlines/section/topic/SPORTS.en_in/Sports?ned=in&hl=en-IN&gl=IN"

    # now call news function with any of these url or BOTH
    news(news_url, 10)
    news(sports_url, 5)
