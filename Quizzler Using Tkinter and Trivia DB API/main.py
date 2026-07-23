
"""This file processes the fetched questions and prepares them for use in the quiz."""

from question_model import Question
from data_dynamic import question_data
from quiz_brain import QuizBrain
from ui import QuizInterface

# question_bank = []
#     question_text = question["question"]
#     question_answer = question["correct_answer"]
#     question_options = question["incorrect_answers"] + [question["correct_answer"]]
#     new_question = Question(question_text, question_answer, question_options)
#     question_bank.append(new_question)

# list comprehension
question_bank = [
    Question(
        question["question"],
        question["correct_answer"],
        question["incorrect_answers"] + [question["correct_answer"]]
    )
    for question in question_data
]

quiz = QuizBrain(question_bank)
quiz_ui = QuizInterface(quiz)
