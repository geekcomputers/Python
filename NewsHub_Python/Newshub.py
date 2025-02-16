import tkinter as tk
import requests
from PIL import Image, ImageTk

# Defining global variables 
Content = ""
News_run = ""

def news():
    global Content
    ApiKey = "Your_Api_Key" # Get your API key 
    url = f"https://newsapi.org/v2/everything?q={Content}&from=2025-01-27&sortBy=popularity&apiKey={ApiKey}"
    response = requests.get(url)
    result = response.json()
    # Create a formatted string from the result
    news_str = ""
    for article in result['articles']:
        news_str += f"Title: {article['title']}\nDescription: {article['description']}\n\n"
    return news_str

def save():
    global Content, label
    Content = str(search.get())
    news_content = news()
    label.config(text=news_content)

window = tk.Tk()
window.geometry("800x600")
window.title("News_App")

# Label for displaying news
label = tk.Label(window, text="", wraplength=750, justify="left")
label.place(x=20, y=100)

# Title label
title_label = tk.Label(window, text="NewsHub", justify="left", font=("Helvetica", 50))
title_label.place(x=10, y=10)

# Display image
img = Image.open("D:\\Downloads\\img.png")
photo_img = ImageTk.PhotoImage(img)
my_img = tk.Label(window, image=photo_img, justify="right")
my_img.place(x=850, y=0)

# Keep a reference to the image to avoid garbage collection
photo_img.image = photo_img

# Search entry field
search = tk.Entry(window, font=("Arial", 20))
search.place(x=300, y=40)

# Search button
find = tk.Button(window, borderwidth=3, cursor="hand2", text="Search", highlightbackground="black", command=save)
find.place(x=650, y=40)

window.mainloop()