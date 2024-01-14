import pyautogui
from time import sleep

# Do you want to include the message counter?
# make a class of it.

# Can make a broswer session open and navigating to web.whatsapp
# os dependencies and browser dependencies and default browser if none
# also check for whatsapp on the system


def send_message(message):
    pyautogui.write(message)
    pyautogui.press("enter")


def send_repeatedly(message, repetitions, delay):
    count = 1
    try:
        for _ in range(repetitions):
            send_message(f"Message {count}: {message}")
            sleep(delay)
            count += 1
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")


if __name__ == "__main__":
    try:
        user_message = input("Enter the message you want to send: ")
        repetitions = int(input("Enter the number of repetitions: "))
        delay = float(input("Enter the delay between messages (in seconds): "))

        sleep(5)
        send_repeatedly(user_message, repetitions, delay)

    except ValueError:
        print("Invalid input. Please enter a valid number.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
