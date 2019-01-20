import random
userGuess = int(input("Enter your guessed no. b/w 0-100:"))
a = comGuess = random.randint(0,100)


while true:
comGuess=random.randint(0,100)
if userGuess>comGuess:
    print("Guess Higher")
     comGuess = random.randint(a,100)
     a++

elif userGuess < comGuess:
    print("Guess Lower")
     comGuess = random.randint(0,a)
     a++

else :
    print ("Guessed Corectly")
    break
