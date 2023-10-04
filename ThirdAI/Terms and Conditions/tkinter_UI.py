from tkinter import *
import tkinter as tk
from tkinter.font import Font
from tkinter import messagebox
from tkinter import filedialog
from thirdai import licensing, neural_db as ndb

licensing.activate("1FB7DD-CAC3EC-832A67-84208D-C4E39E-V3")
db = ndb.NeuralDB(user_id="my_user")

root = Tk()
root.geometry("600x500")
root.title('ThirdAI - T&C')

path = []


def customsize(sizeup):
    return Font(size=sizeup)


def clear_all():
    query_entry.delete(0, tk.END)
    text_box.delete(1.0, tk.END)
    path.clear()


def training():
    insertable_docs = []
    value = path[0]
    pdf_files = value

    pdf_doc = ndb.PDF(value)
    insertable_docs.append(pdf_doc)

    print(insertable_docs)

    source_ids = db.insert(insertable_docs, train=True)

    def show_training_done_message():
        messagebox.showinfo("Training Complete", "Training is done!")

    show_training_done_message()


def processing():
    question = query_entry.get()
    search_results = db.search(
        query=question,
        top_k=2,
        on_error=lambda error_msg: print(f"Error! {error_msg}"))

    output = ""
    for result in search_results:
        output += result.text + "\n"

    def process_data(output_data):
        output_window = tk.Toplevel(root)
        output_window.title("Output Data")
        output_window.geometry("500x500")

        output_text = tk.Text(output_window, wrap=tk.WORD, width=50, height=50)
        output_text.pack(padx=10, pady=10)
        output_text.insert(tk.END, output_data)

    process_data(output)


def fileinput():
    global path
    win = Tk()
    win.withdraw()
    file_type = dict(defaultextension=".pdf", filetypes=[("pdf file", "*.pdf")])
    file_path = filedialog.askopenfilename(**file_type)
    print(file_path)
    file = file_path.split("/")
    print(file[-1])
    path.append(file[-1])
    print(path)
    text_box.insert(INSERT, file[-1])


menu = Label(root, text="Terms & Conditions", font=customsize(30), fg='black', highlightthickness=2,
             highlightbackground="red")
menu.place(x=125, y=10)

insert_button = Button(root, text="Insert File!", font=15, fg='black', bg="grey", width=10, command=fileinput)
insert_button.place(x=245, y=100)

text_box = tk.Text(root, wrap=tk.WORD, width=30, height=1)
text_box.place(x=165, y=150)

training = Button(root, text="Training", font=15, fg='black', bg="grey", width=10, command=training)
training.place(x=245, y=195)

query = Label(root, text="Query", font=customsize(20), fg='black')
query.place(x=255, y=255)

query_entry = tk.Entry(root, font=customsize(20), width=30)
query_entry.place(x=70, y=300)

processing = Button(root, text="Processing", font=15, fg='black', bg="grey", width=10, command=processing)
processing.place(x=245, y=355)

clear = Button(root, text="Clear", font=15, fg='black', bg="grey", width=10, command=clear_all)
clear.place(x=245, y=405)

root.mainloop()
