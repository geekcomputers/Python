import time
import random

sentences = [
    "Python is a powerful and easy to learn programming language.",
    "Open source software encourages collaboration and innovation.",
    "Practice coding every day to improve your programming skills.",
    "Automation can save time and reduce repetitive work.",
    "Git and GitHub are essential tools for developers.",
    "Clean code is easier to read and maintain.",
    "Debugging is an important part of the development process.",
]

def calculate_wpm(text, elapsed_time):
    words = len(text.split())
    minutes = elapsed_time / 60
    return round(words / minutes)

def typing_test():
    sentence = random.choice(sentences)

    print("\nTyping Speed Test\n")
    print("Type the following sentence:\n")
    print(sentence)
    input("\nPress ENTER when you are ready...")

    start_time = time.time()

    typed = input("\nStart typing:\n")

    end_time = time.time()

    elapsed_time = end_time - start_time

    correct_chars = 0
    for i in range(min(len(sentence), len(typed))):
        if sentence[i] == typed[i]:
            correct_chars += 1

    accuracy = (correct_chars / len(sentence)) * 100

    wpm = calculate_wpm(typed, elapsed_time)

    print("\nResults")
    print("-------")
    print(f"Time taken: {round(elapsed_time,2)} seconds")
    print(f"Typing speed: {wpm} WPM")
    print(f"Accuracy: {round(accuracy,2)} %")

if __name__ == "__main__":
    typing_test()
