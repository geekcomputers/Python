# Tik-tak game


board = ["anything", 1, 2, 3, 4, 5, 6, 7, 8, 9]
switch = "p1"
j = 9
print("\n\t\t\tTIK-TAC-TOE")


def print_board():
    """
    Prints the tic-tac-toe board.

    Parameters: None
    Returns: None

        |     | 
     1  | 2   | 3 
    ____|_____|____   <-- Horizontal line (top) of board. The
    underscores are to make it easier to read the numbers on the board. The spaces between each number are for visual clarity only, and do not affect
    gameplay in any way. They can be removed without affecting game play at all, but I decided to leave them in because they look nicer and make it easier
    for humans to visualize where each space is on the board when playing against another human player or against an AI opponent that has a limited search
    depth (i.e., minimax). This is especially important if you want your program's search depth > 1! If you remove these spaces from this docstring, then
    remove them from print_board() as well! :)

        |     | 

     4  | 5   | 6      <-- Vertical line separating first row of numbers from second row of
    numbers (left side). Same deal here with spacing; see above comment about why I left these in... :P

    ____|_____|____    <-- Horizontal line separating
    second row of
    """
    # import os
    # os.system('cls')
    print("\n\n")
    print("    |     |")
    print("", board[1], " | ", board[2], " | ", board[3])
    print("____|_____|____")
    print("    |     |")
    print("", board[4], " | ", board[5], " | ", board[6])
    print("____|_____|____")
    print("    |     |")
    print("", board[7], " | ", board[8], " | ", board[9])
    print("    |     |")


def enter_number(p1_sign, p2_sign):
    """
    This function is used to enter the number for the game.
    """
    global switch
    global j
    k = 9
    while (j):
        if k == 0:
            break

        if switch == "p1":
            p1_input = int(input("\nplayer 1 :- "))
            if p1_input <= 0:
                print("chose number from given board")
            else:
                for e in range(1, 10):
                    if board[e] == p1_input:
                        board[e] = p1_sign
                        print_board()
                        c = checkwin()
                        if c == 1:
                            print("\n\n Congratulation ! player 1 win ")
                            return

                        switch = "p2"
                        j -= 1
                        k -= 1
                        if k == 0:
                            print("\n\nGame is over")
                            break

        if k == 0:
            break

        if switch == "p2":
            p2_input = int(input("\nplayer 2 :- "))
            if p2_input <= 0:
                print("chose number from given board")
                # return
            else:
                for e in range(1, 10):
                    if board[e] == p2_input:
                        board[e] = p2_sign
                        print_board()
                        w = checkwin()
                        if w == 1:
                            print("\n\n Congratulation ! player 2 win")
                            return

                        switch = "p1"
                        j -= 1
                        k -= 1


def checkwin():
    """
    This function checks if the player has won or not.
    It takes no arguments and returns 1 if the player wins, 0 otherwise.
    """
    if board[1] == board[2] == board[3]:

        return 1
    elif board[4] == board[5] == board[6]:

        return 1
    elif board[7] == board[8] == board[9]:

        return 1
    elif board[1] == board[4] == board[7]:

        return 1

    elif board[2] == board[5] == board[8]:

        return 1
    elif board[3] == board[6] == board[9]:

        return 1
    elif board[1] == board[5] == board[9]:

        return 1
    elif board[3] == board[5] == board[7]:

        return 1
    else:
        print("\n\nGame continue")


def play():
    """
    This function is used to play the game.
    It takes two inputs from user i.e p1_sign and p2_sign
    and then it prints the board after every move of player
    1 and 2 respectively.
    """
    print_board()
    p1_sign = input("\n\nplayer 1 chose your sign [0/x] = ")
    p2_sign = input("player 2 chose your sign [0/x] = ")
    enter_number(p1_sign, p2_sign)
    print("\n\n\t\t\tDeveloped By :- UTKARSH MATHUR")


if __name__ == "__main__":
    play()
