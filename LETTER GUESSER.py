import random
import string

ABCS = string.ascii_lowercase
ABCS = list(ABCS)

play = True

compChosse = random.choice(ABCS)

print(":guess the letter (only 10 guesses):")
userInput = input("guess:")

failed = 10

while failed > 0:
	if userInput == compChosse:
		print("---------->")
		print("You are correct!")
		print("---------->")
		print("Your guesses: " + str(10 - failed))
		break

	elif userInput != compChosse:
		failed = failed - 1
	
		print(":no your wrong: " + "left: " + str(failed))
	
		userInput = input("guess:")

	if failed == 0:
		print("out of guesses")
