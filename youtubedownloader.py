from tkinter import *
from tkinter import filedialog, messagebox
from threading import Thread
from pytube import YouTube


def threading():
    # Call work function
    t1 = Thread(target=download)
    t1.start()

def download():
    try:
        url = YouTube(str(url_box.get()))
        video = url.streams.first()
        filename = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        video.download(filename=filename)
        messagebox.showinfo('', 'Download completed!')
    except Exception as e:
        messagebox.showerror("Error", "An error occurred while downloading the video.")


root = Tk()
root.title('YouTube Downloader')
root.geometry('780x500+200+200')
root.configure(bg='olivedrab1')
root.resizable(False, False)

# Label widgets
introlable = Label(
    root,
    text='YouTube Video Downloader',
    width=30,
    relief='ridge',
    bd=4,
    font=('chiller', 26, 'italic bold'),
    fg='red')
introlable.place(x=35, y=20)

Label(
    root, 
    text='Enter YouTube Link', 
    font=('sans-serif', 16), 
    bg='olivedrab1'
    ).place(x=40, y=150)

url_box = Entry(
    root, 
    font=('arial', 30), 
    width=30
    )
url_box.place(x=40, y=180)

btn = Button(
    root, 
    text='DOWNLOAD', 
    font=('sans-serif', 25), 
    command=threading
    )
btn.place(x=270, y=240)

root.mainloop()
