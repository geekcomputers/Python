import random

a = comGuess = random.randint(0,100)


while True:
	userGuess = int(input("Enter your guessed no. b/w 0-100:"))
	if userGuess < comGuess:
	    print("Guess Higher")
	    comGuess = random.randint(a,100)
	    a += 1

	elif userGuess > comGuess:
	    print("Guess Lower")
	    comGuess = random.randint(0,a)
	    a += 1

	else :
	    print ("Guessed Corectly")
	    break
