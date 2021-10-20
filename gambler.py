import random
from sys import argv

stake = int(argv[1])
goals = int(argv[2])
trials = int(argv[3])

wins = 0
bets = 0

for i in range(trials):
    cash = stake
    while cash > 0 and cash < goals:
        bets += 1
        if random.randrange(0, 2) == 0:
            cash += 1
        else:
            cash -= 1
    if cash == goals:
        wins += 1
print("Your won: " + str(100 * wins // trials) + "$")
print("Your bets: " + str(bets // trials))
