from tkinter import *
from threading import Thread
from tkinter import messagebox
from pytube import YouTube


def threading():
    # Call work function
    t1=Thread(target=download)
    t1.start()

def download():
    url = YouTube(str(url_box.get()))
    video = url.streams.first()
    video.download()    
    messagebox.showinfo('', 'Download completed!')

root =Tk()
root.title('YouTube Downloader')
root.geometry('780x500+200+200')
#root.iconbitmap('youtube.ico')
root.configure(bg='olivedrab1')
root.resizable(False,False)

# Label widgets
introlable = Label(
    root,
    text='YouTube Video Downloader',
    width=30,
    relief='ridge',
    bd=4,
    font=('chiller',26,'italic bold'),
    fg='red')
introlable.place(x=35,y=20)

Label(
    root, 
    text='Enter YouTube Link', 
    font=('sans-serif', 16), 
    bg='olivedrab1'
    ).place(x=40, y=150)

# DownloadingSizeLabel = Label(
#     root,
#     text='Total Size: ',
#     font=('arial',15,'italic bold'),
#     bg='olivedrab1'
#     )
# DownloadingSizeLabel.place(x=500,y=240)

# DownloadingLabel = Label(
#     root,
#     text='Recieved Size: ',
#     font=('arial',15,'italic bold'),
#     bg='olivedrab1'
#     )
# DownloadingLabel.place(x=500,y=290)

# DownloadingTime = Label(
#     root,
#     text='Time Left : ',
#     font=('arial',15,'italic bold'),
#     bg='olivedrab1'
#     )
# DownloadingTime.place(x=500,y=340)

# DownloadingSizeLabelResult = Label(
#     root,
#     text=' ',
#     font=('arial',15,'italic bold'),
#     bg='olivedrab1'
#     )
# DownloadingSizeLabelResult.place(x=650,y=240)

# DownloadingLabelResult = Label(
#     root,
#     text=' ',
#     font=('arial',15,'italic bold'),
#     bg='olivedrab1'
#     )
# DownloadingLabelResult.place(x=650,y=290)

# Entry widgets
url_box = Entry(
    root, 
    font=('arial',30), 
    width=30
    )
url_box.place(x=40, y=180)

# Button Widgets
btn = Button(
    root, 
    text='DOWNLOAD', 
    font=('sans-serif', 25), 
    command=threading
    )
btn.place(x=270, y=240)

root.mainloop()
