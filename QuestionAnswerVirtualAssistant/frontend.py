from tkinter import *
from tkinter import messagebox
import backend


def index_question_answer():
    # for this, we are separating question and answer by "_"
    question_answer = index_question_answer_entry.get()
    question, answer = question_answer.split("_")
    print(question)
    print(answer)
    va = backend.QuestionAnswerVirtualAssistant()
    print(va.index_question_answer(question, answer))

def provide_answer():
    term = provide_answer_entry.get()
    va = backend.QuestionAnswerVirtualAssistant()
    print(va.provide_answer(term))

if __name__ == "__main__":
    root = Tk()
    root.title("Knowledge base")
    root.geometry('300x300')

    index_question_answer_label = Label(root, text="Add question:")
    index_question_answer_label.pack()
    index_question_answer_entry = Entry(root)
    index_question_answer_entry.pack()

    index_question_answer_button = Button(root, text="add", command=index_question_answer)
    index_question_answer_button.pack()

    provide_answer_label = Label(root, text="User Input:")
    provide_answer_label.pack()
    provide_answer_entry = Entry(root)
    provide_answer_entry.pack()

    search_term_button = Button(root, text="ask", command=provide_answer)
    search_term_button.pack()

    root.mainloop()