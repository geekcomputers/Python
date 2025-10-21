
"""This file manages the graphical user interface of the quiz, using Tkinter to display questions, answer options, and the score to the user."""

from tkinter import *
from quiz_brain import QuizBrain
from data_dynamic import error_message

# Normal screen
BACKGROUND = "#608BC1"
CANVAS = "#CBDCEB"
TEXT = "#133E87"

# If answer is right
R_BACKGROUND = "#859F3D"
R_CANVAS = "#F6FCDF"
R_TEXT = "#31511E"

# If answer is wrong
W_BACKGROUND = "#C63C51"
W_CANVAS = "#D95F59"
W_TEXT = "#522258"

FONT = ("Lucida sans", 20)

class QuizInterface:

    def __init__(self, quiz_brain: QuizBrain):
        self.quiz = quiz_brain
        self.window = Tk()
        self.window.title("Quizzler")
        self.window.config(padx=20, pady=20, bg=BACKGROUND)

        self.score_label = Label(text="Score: 0", fg="white", bg=BACKGROUND, font=("Lucida sans", 15, "bold"))
        self.score_label.grid(row=0, column=1)

        self.canvas = Canvas(width=1000, height=550, bg=CANVAS)
        self.question_text = self.canvas.create_text(
            500, 100, width=800, text="Some question text", fill=TEXT, font=FONT, anchor="center", justify="center"
        )
        self.canvas.grid(row=1, column=0, columnspan=2, pady=50)

        self.opt_selected = IntVar()
        self.options = self.create_radio_buttons()

        self.submit_button = Button(
            text="Submit", command=self.submit_answer, fg=TEXT, font=FONT
        )
        self.submit_button.grid(row=3, column=0, columnspan=2)

        if error_message:
            self.display_error_message(error_message)
        else:
            self.get_next_question()

        self.window.mainloop()

    def create_radio_buttons(self):
        radio_buttons = []
        y_position = 230
        for i in range(4):
            radio_button = Radiobutton(
                self.canvas, text="", variable=self.opt_selected, value=i + 1, font=FONT, bg=CANVAS, anchor="w", 
                justify="left", fg=TEXT, wraplength=900
            )
            radio_buttons.append(radio_button)
            self.canvas.create_window(50, y_position, window=radio_button, anchor="w")
            y_position += 65
        return radio_buttons

    def get_next_question(self):
        if self.quiz.still_has_questions():
            self.opt_selected.set(0)  # Reset selection
            q_text = self.quiz.next_question()
            self.canvas.itemconfig(self.question_text, text=q_text)
            self.canvas.config(bg=CANVAS)
            self.window.config(bg=BACKGROUND)
            for option in self.options:
                option.config(bg=CANVAS, fg=TEXT)
            self.display_options()
            self.score_label.config(bg=BACKGROUND, text=f"Score: {self.quiz.score}")
            self.canvas.itemconfig(self.question_text, fill=TEXT)
        else:
            self.display_result()

    def display_options(self):
        current_options = self.quiz.current_question.options
        for i, option in enumerate(current_options):
            self.options[i].config(text=option)

    def submit_answer(self):
        selected_option_index = self.opt_selected.get() - 1
        if selected_option_index >= 0:
            user_answer = self.quiz.current_question.options[selected_option_index]
            self.quiz.check_answer(user_answer)

            if self.quiz.check_answer(user_answer):
                self.quiz.score += 1
                self.canvas.config(bg=R_CANVAS)
                self.window.config(bg=R_BACKGROUND)
                for option in self.options:
                    option.config(bg=R_CANVAS, fg=R_TEXT)
                self.canvas.itemconfig(self.question_text, fill=R_TEXT)
                self.score_label.config(bg=R_BACKGROUND)
            else:
                self.canvas.config(bg=W_CANVAS)
                self.window.config(bg=W_BACKGROUND)
                for option in self.options:
                    option.config(bg=W_CANVAS, fg=W_TEXT)
                self.canvas.itemconfig(self.question_text, fill=W_TEXT)
                self.score_label.config(bg=W_BACKGROUND)

            self.window.after(1000, self.get_next_question)

    def display_result(self):
        for option in self.options:
            option.config(bg=CANVAS, fg=TEXT)
            option.destroy()

        if self.quiz.score <= 3:
            self.result_text = f"You've completed the quiz!\nYour final score: {self.quiz.score}/{self.quiz.question_number}\nBetter luck next time! Keep practicing!"
        elif self.quiz.score <= 6:
            self.result_text = f"You've completed the quiz!\nYour final score: {self.quiz.score}/{self.quiz.question_number}\nGood job! You're getting better!"
        elif self.quiz.score <= 8:
            self.result_text = f"You've completed the quiz!\nYour final score: {self.quiz.score}/{self.quiz.question_number}\nGreat work! You're almost there!"
        else:
            self.result_text = f"You've completed the quiz!\nYour final score: {self.quiz.score}/{self.quiz.question_number}\nExcellent! You're a Quiz Master!"

        self.score_label.config(bg=BACKGROUND, text=f"Score: {self.quiz.score}")
        self.canvas.config(bg=CANVAS)
        self.window.config(bg=BACKGROUND)
        self.canvas.itemconfig(self.question_text, fill=TEXT)
        self.score_label.config(bg=BACKGROUND)

        self.canvas.itemconfig(self.question_text, text=self.result_text)
        self.canvas.coords(self.question_text, 500, 225)  # Centered position
        self.submit_button.config(state="disabled")

    def display_error_message(self, message):
        for option in self.options:
            option.config(bg=CANVAS, fg=TEXT)
            option.destroy()

        self.canvas.itemconfig(self.question_text, text=message)
        self.canvas.coords(self.question_text, 500, 225)  # Centered position
        self.submit_button.config(state="disabled")
