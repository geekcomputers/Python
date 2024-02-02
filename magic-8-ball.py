import random
import time

def magic_8_ball():
    answers = [
        "Yes",
        "No",
        "Ask again later",
        "Cannot predict now",
        "Don't count on it",
        "Most likely",
        "Outlook not so good",
        "Reply hazy, try again"
    ]

    print("Welcome to the Magic 8-Ball!")
    time.sleep(1)

    while True:
        question = input("Ask me a yes-or-no question (or type 'exit' to end): ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break

        print("Shaking the Magic 8-Ball...")
        time.sleep(2)
        answer = random.choice(answers)
        print(f"The Magic 8-Ball says: {answer}\n")

if __name__ == "__main__":
    magic_8_ball()
