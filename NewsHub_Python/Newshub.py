import tkinter as Tk 
from PIL import Image, ImageTk

# Defining global variables 
content = ""
news_content = ""

def fetch_news():
    global content
    api_key = "Your_Api_Key"
    url = f"https://newsapi.org/v2/everything?q={content}&from=2025-01-27&sortBy=popularity&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        result = response.json()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

    news_str = ""
    for article in result.get('articles', []):
        news_str += f"Title: {article['title']}\nDescription: {article['description']}\n\n"
    return news_str

def save():
    global content, label
    content = search.get()
    news_str = fetch_news()
    label.config(text=news_str)

window = Tk.Tk()
window.geometry("800x600")
window.title("News App")

# Label for displaying news
label = Tk.Label(window, text="", wraplength=750, justify="left")
label.place(x=20, y=100)

# Title label
title_label = Tk.Label(window, text="NewsHub", justify="left", font=("Helvetica", 50))
title_label.place(x=10, y=10)

# Display image
try:
    img = Image.open("D:\\Downloads\\img.png")
    photo_img = ImageTk.PhotoImage(img)
    my_img = Tk.Label(window, image=photo_img, justify="right")
    my_img.place(x=850, y=0)
    # Keep a reference to the image to avoid garbage collection
    photo_img.image = photo_img
except Exception as e:
    print(f"Error loading image: {e}")

# Search entry field
search = Tk.Entry(window, font=("Arial", 20))
search.place(x=300, y=40)

# Search button
find = Tk.Button(window, borderwidth=3, cursor="hand2", text="Search", highlightbackground="black", command=save)
find.place(x=650, y=40)

window.mainloop()
