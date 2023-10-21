# Importing the required libraries.
import pyshorteners

# Taking input from the user.
url = input("Enter URL: ")

# Creating an instance of the pyshorteners library.
shortener = pyshorteners.Shortener()

# Shortening the URL using TinyURL.
shortened_URL = shortener.tinyurl.short(url)

# Displaying the shortened URL.
print(f"Shortened URL: {shortened_URL}")
