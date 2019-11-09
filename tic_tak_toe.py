import itertools

from colorama import Fore, Style, init

init()


def win(current_game):
    def all_same(l):
        if l.count(l[0]) == len(l) and l[0] != 0:
            return True
        else:
            return False

    # horizontal
    for row in game:
        if all_same(row):
            print("Player {} is the winner horizontally!".format(row[0]))
            print("Congrats")
        return True

    # vertical
    for col in range(len(game)):
        check = []
        for row in game:
            check.append(row[col])
        if all_same(check):
            print("Player {} is the winner vertically!".format(check[0]))
            return True

    # / diagonal
    diags = []
    for idx, reverse_idx in enumerate(reversed(range(len(game)))):
        diags.append(game[idx][reverse_idx])
    if all_same(diags):
        print("Player {} is the winner diagonally(/)!".format(diags[0]))
        return True

    # \ diagonal
    diags = []
    for idx in range(len(game)):
        diags.append(game[idx][idx])

    if all_same(diags):
        print("Player {diags[0]} has won Diagonally (\\)")
        return True

    return False


def game_board(game_map, player=0, row=0, column=0, just_display=False):
    try:

        if game_map[row][column] != 0:
            print("This space is occupied, try another!")
            return False

        print("   " + "  ".join([str(i) for i in range(len(game_map))]))
        if not just_display:
            game_map[row][column] = player

        for count, row in enumerate(game_map):
            colored_row = ""
            for item in row:
                if item == 0:
                    colored_row += "   "
                elif item == 1:
                    colored_row += Fore.GREEN + " X " + Style.RESET_ALL
                elif item == 2:
                    colored_row += Fore.MAGENTA + " O " + Style.RESET_ALL
            print(count, colored_row)

        return game_map

    except IndexError:
        print("Did you attempt to play a row or column outside the range of 0,1 or 2? (IndexError)")
        return False
    except Exception as e:
        print(str(e))
        return False


play = True
Players = [1, 2]

while play:
    game_size = int(input("What size game TicTacToe? "))
    game = [[0 for i in range(game_size)] for i in range(game_size)]

    game_won = False
    player_cycle = itertools.cycle([1, 2])
    game_board(game, just_display=True)
    while not game_won:
        current_player = next(player_cycle)
        Played = False

        while not Played:
            print("Player: {}".format(current_player))
            row_choice = int(input("Which row? "))
            column_choice = int(input("Which column? "))
            Played = game_board(game, player=current_player, row=row_choice, column=column_choice)

        if win(game):
            game_won = True
            again = input("The game is over,would you like to play again? (y/n) ")
            if again.lower() == "y":
                print("restarting!")
            elif again.lower() == "n":
                print("Byeeeee!!!")
                play = False
            else:
                print("not a valid answer!!")
                play = False
