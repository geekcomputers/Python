import time
import random
print("Hello Welcome To The Guess Game!")
time.sleep(1)
print("I\'m Geek! What's Your Name?")
name=input()
time.sleep(1)
print(f"Okay {name} Let's Begin The Guessing Game!")
a = comGuess = random.randint(0, 100)  # a and comGuess is initialised with a random number between 0 and 100
int count=0
while True:  # loop will run until encountered with the break statement(user enters the right answer)
    userGuess = int(input("Enter your guessed no. b/w 0-100:"))  # user input for guessing the number
    if userGuess < comGuess:  # if number guessed by user is lesser than the random number than the user is told to guess higher and the random number comGuess is changed to a new random number between a and 100
        print("Guess Higher")
        comGuess = random.randint(a, 100)
        a += 1
        count++

    elif userGuess > comGuess:  # if number guessed by user is greater than the random number than the user is told to guess lower and the random number comGuess is changed to a new random number between 0 and a
        print("Guess Lower")
        comGuess = random.randint(0, a)
        a += 1
        count++

   
    elif userGuess == comGuess and count==0 :#the player needs a special reward for perfect guess in the first try ;-)
        print("Bravo! Legendary Guess!") 
    
    else:#Well, A Congratulations Message For Guessing Correctly!
        print("Congratulations, You Guessed It Correctly!")
