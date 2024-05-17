from tkinter import *
from tkinter import messagebox
import backend


def add_document():
    document = add_documents_entry.get()
    se = backend.SearchEngine()
    print(se.index_document(document))

def find_term():
    term = find_term_entry.get()
    se = backend.SearchEngine()
    print(se.find_documents(term))

if __name__ == "__main__":
    root = Tk()
    root.title("Registration Form")
    root.geometry('300x300')

    add_documents_label = Label(root, text="Add Document:")
    add_documents_label.pack()
    add_documents_entry = Entry(root)
    add_documents_entry.pack()

    add_document_button = Button(root, text="add", command=add_document)
    add_document_button.pack()

    find_term_label = Label(root, text="Input term to search:")
    find_term_label.pack()
    find_term_entry = Entry(root)
    find_term_entry.pack()

    search_term_button = Button(root, text="search", command=find_term)
    search_term_button.pack()

    root.mainloop()