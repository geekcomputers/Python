def run_quiz(questions):
    score = 0

    for question, options, correct_option in questions:
        print("\n" + question)
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        answer = input("\nEnter the number of the correct answer: ")

        if answer.isdigit() and int(answer) == correct_option:
            print("Correct!")
            score += 1
        else:
            print("Sorry, that's incorrect.")

    print(f"\nYour final score is: {score}/{len(questions)}")

if __name__ == "__main__":
    quiz_questions = [
        ("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"], 2),
        ("Which planet is known as the Red Planet?", ["Earth", "Mars", "Jupiter", "Saturn"], 2),
        ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Arctic", "Pacific"], 4),
    ]

    run_quiz(quiz_questions)
