import random
import pandas as pd

# Taking players data
players = {}  # stores players name their locations
isReady = {}
current_loc = 1  # vaiable for iterating location

# creating board
board = [
    [100, 99, 98, "S", 96, "S", 94, 93, 92, 91],
    [81, 82, 83, 84, 85, 86, "L", "S", 89, 90],
    [80, 79, 78, 77, 76, 75, 74, 73, 72, "L"],
    [61, 62, "S", 64, 65, 66, 67, 68, 69, 70],
    [60, 59, 58, 57, 56, 55, 54, 53, 52, 51],
    [41, 42, 43, 44, 45, 46, 47, "S", 49, "L"],
    ["L", 39, 38, 37, "S", 35, 34, 33, "S", 31],
    [21, 22, 23, 24, 25, 26, 27, "L", 29, 30],
    ["L", 19, 18, 17, 16, 15, 14, 13, 12, 11],
    [1, 2, 3, "L", 5, 6, 7, "L", 9, 10],
]

df = pd.DataFrame(board)

styled_df = df.style \
    .set_properties(**{'background-color': 'lightblue', 'color': 'black'}) \
    .set_table_styles([{
        'selector': 'td',
        'props': [
            ('padding', '20px'),
        ]
    }])

# DataFrame as HTML
html_output = styled_df.render()

# Save the HTML output
with open('styled_df_output.html', 'w') as f:
    f.write(html_output)

print("HTML output saved to 'styled_df_output.html'")

imp = True

# players input function
def player_input():
    global players
    global current_loc
    global isReady

    x = True
    while x:
        player_num = int(input("Enter the number of players: "))
        if player_num > 0:
            for i in range(player_num):
                name = input(f"Enter player {i+1} name: ")
                players[name] = current_loc
                isReady[name] = False
            x = False
            play()  # play funtion call

        else:
            print("Number of player cannot be zero")
            print()


# Dice roll method
def roll():
    # print(players)
    return random.randrange(1, 7)


# play method
def play():
    global players
    global isReady
    global imp

    print("/"*20)
    print("1 -> roll the dice (or enter)")
    print("2 -> start new game")
    print("3 -> exit the game")
    print("/"*20)

    while imp:
        for i in players:
            n = input("{}'s turn: ".format(i)) or 1
            n = int(n)

            if players[i] < 100:
                if n == 1:
                    temp1 = roll()
                    print(f"you got {temp1}")
                    print("")

                    if isReady[i] == False and temp1 == 6:
                        isReady[i] = True

                    if isReady[i]:
                        looproll = temp1
                        counter_6 = 0
                        while looproll == 6:
                            counter_6 += 1
                            looproll = roll()
                            temp1 += looproll
                            print(f"you got {looproll} ")
                            if counter_6 == 3 :
                                temp1 -= 18
                                print("Three consectutives 6 got cancelled")
                            print("")
                        # print(temp1)
                        if (players[i] + temp1) > 100:
                            pass
                        elif (players[i] + temp1) < 100:
                            players[i] += temp1
                            players[i] = move(players[i], i)
                        elif (players[i] + temp1) == 100:
                            print(f"congrats {i} you won !!!")
                            imp = False
                            return

                    print(f"you are at position {players[i]}")
                    print('-'*20)

                elif n == 2:
                    players = {}  # stores player ans their locations
                    isReady = {}
                    current_loc = 0  # vaiable for iterating location
                    player_input()

                elif n == 3:
                    print("Bye Bye")
                    imp = False

                else:
                    print("pls enter a valid input")


# Move method
def move(a, i):
    global players
    global imp
    temp_loc = players[i]

    if (temp_loc) < 100:
        temp_loc = ladder(temp_loc, i)
        temp_loc = snake(temp_loc, i)

        return temp_loc


# snake bite code
def snake(c, i):
    if (c == 32):
        players[i] = 10
    elif (c == 36):
        players[i] = 6
    elif (c == 48):
        players[i] = 26
    elif (c == 63):
        players[i] = 18
    elif (c == 88):
        players[i] = 24
    elif (c == 95):
        players[i] = 56
    elif (c == 97):
        players[i] = 78
    else:
        return players[i]
    print(f"You got bitten by a snake now you are at {players[i]}")

    return players[i]


# ladder code
def ladder(a, i):
    global players

    if (a == 4):
        players[i] = 14
    elif (a == 8):
        players[i] = 30
    elif (a == 20):
        players[i] = 38
    elif (a == 40):
        players[i] = 42
    elif (a == 28):
        players[i] = 76
    elif (a == 50):
        players[i] = 67
    elif (a == 71):
        players[i] = 92
    elif (a == 87):
        players[i] = 99
    else:
        return players[i]
    print(f"You got a ladder now you are at {players[i]}")

    return players[i]


# while run:
print("/"*40)
print("Welcome to the snake ladder game !!!!!!!")
print("/"*40)


player_input()