# modules for Using of app
from tkinter import Button, Entry, Label, Tk, filedialog, messagebox # Gui Modules 
from threading import Thread # modules for multi threding 
from pytube import YouTube # Module for Youtube service

# this function for mulple code runes at a time 
def threading():
    # Call work function
    t1 = Thread(target=download)
    t1.start()

# this function for Download Youtube video
def download():
    try:
        url = YouTube(str(url_box.get()))
        video = url.streams.first()
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")]
        )
        if filename:  # Check if a filename is selected
            video.download(filename=filename)
            messagebox.showinfo("", "Download completed!")
        else:
            messagebox.showwarning("", "Download cancelled!")
    except Exception:
        messagebox.showerror("Error", "Some Thing Went Wrong!!!\nplease try again")

        
# This code runes on only this file
if __name__=="__main__":
    root = Tk()
    root.title("YouTube Downloader")
    root.geometry("780x500+200+200")
    root.configure(bg="olivedrab1")
    root.resizable(False, False)
    # Label widgets
    introlable = Label(
        root,
        text="YouTube Video Downloader",
        width=30,
        relief="ridge",
        bd=4,
        font=("chiller", 26, "italic bold"),
        fg="red",
    )
    introlable.place(x=35, y=20)

    Label(root, text="Enter YouTube Link", font=("sans-serif", 16), bg="olivedrab1", fg='Black').place(
        x=40, y=150
    )

    # entry box in UI
    url_box = Entry(root, font=("arial", 30), width=30)
    url_box.place(x=40, y=180)

    # download button in UI
    btn = Button(root, text="DOWNLOAD", font=("sans-serif", 25), command=threading)
    btn.place(x=270, y=240)
    root.mainloop()
