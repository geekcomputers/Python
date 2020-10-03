'''
  hey everyone it is a basic game code using random . in this game computer will randomly chose an number from 1 to 100 and players will have 
  to guess that which number it is and game will tell him on every guss whether his/her guess is smaller or bigger than the chosen number. it is 
  a multi player game so it can be played with many players there is no such limitations of user till the size of list. if any one wants to modify 
  this game he/she is most welcomed.
    Thank you
'''

import os
import random

players=[]
score=[]

print("\n\tRandom Number Game\n\nHello Everyone ! it is just a game of chance in which you have to guess a number"
      " from 0 to 100 and computer will tell whether your guess is smaller or bigger than the acctual number chossen by the computer . "
      "the person with less attempts in guessing the number will be winner .")
x=input()
os.system('cls')

n=int(input("Enter number of players : "))
print()

for i in range(0,n):
    name=input("Enter name of player : ")
    players.append(name)

os.system('cls')

for i in range(0,n):
    orignum=random.randint(1,100)
    print(players[i],"your turn :",end="\n\n")
    count=0
    while True :
        ch=int(input("Please enter your guess : "))
        if ch>orignum:
            print("no! number is smaller...")
            count+=1
        elif ch==orignum:
            print("\n\n\tcongrats you won")
            break
        else :
            print("nope ! number is large dude...")
            count+=1
    print("    you have taken", count+1,"attempts")
    x=input()
    score.append(count+1)
    os.system('cls')
print("players :\n")
for i in range(0,n):
    print(players[i],"-",score[i])

print("\n\nwinner is :\n")
for i in range(0,n):
    if score[i]==min(score):
        print(players[i])
x=input()
