import random

# a python program for tic-tac-toe game
# module intro for introduction
# module show_board for values
# module playgame


def introduction():
    print("Hello this a sample tic tac toe game")
    print("It will rotate turns between players one and two")
    print("While 3,3 would be the bottom right.")
    print("Player 1 is X and Player 2 is O")


def draw_board(board):
    print("    |    |")
    print(f"  {board[7]} | {board[8]}  | {board[9]}")
    print("    |    |")
    print("-------------")
    print("    |    |")
    print(f"  {board[4]} | {board[5]}  | {board[6]}")
    print("    |    |")
    print("-------------")
    print("    |    |")
    print(f"  {board[1]} | {board[2]}  | {board[3]}")
    print("    |    |")


def input_player_letter():
    # Lets the player type witch letter they want to be.
    # Returns a list with the player's letter as the first item, and the computer's letter as the second.
    letter = ""
    while letter not in {"X", "O"}:
        print("Do you want to be X or O? ")
        letter = input("> ").upper()

    # the first element in the list is the player’s letter, the second is the computer's letter.
    return ["X", "O"] if letter == "X" else ["O", "X"]


def frist_player():
    guess = random.randint(0, 1)
    return "Computer" if guess == 0 else "Player"


def play_again():
    print("Do you want to play again? (y/n)")
    return input().lower().startswith("y")


def make_move(board, letter, move):
    board[move] = letter


def is_winner(bo, le):
    # Given a board and a player’s letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don’t have to type as much.
    return (
        (bo[7] == le and bo[8] == le and bo[9] == le)
        or (bo[4] == le and bo[5] == le and bo[6] == le)
        or (bo[1] == le and bo[2] == le and bo[3] == le)
        or (bo[7] == le and bo[4] == le and bo[1] == le)
        or (bo[8] == le and bo[5] == le and bo[2] == le)
        or (bo[9] == le and bo[6] == le and bo[3] == le)
        or (bo[7] == le and bo[5] == le and bo[3] == le)
        or (bo[9] == le and bo[5] == le and bo[1] == le)
    )


def get_board_copy(board):
    return list(board)


def is_space_free(board, move):
    return board[move] == " "


def get_player_move(board):
    # Let the player type in their move
    move = " "
    while move not in "1 2 3 4 5 6 7 8 9".split() or not is_space_free(
        board, int(move)
    ):
        print("What is your next move? (1-9)")
        move = input()
    return int(move)


def choose_random_move_from_list(board, moveslist):
    possible_moves = [i for i in moveslist if is_space_free(board, i)]
    return random.choice(possible_moves) if possible_moves else None


def get_computer_move(board, computer_letter):
    player_letter = "O" if computer_letter == "X" else "X"
    for i in range(1, 10):
        copy = get_board_copy(board)
        if is_space_free(copy, i):
            make_move(copy, computer_letter, i)
            if is_winner(copy, computer_letter):
                return i

    for i in range(1, 10):
        copy = get_board_copy(board)
        if is_space_free(copy, i):
            make_move(copy, player_letter, i)
            if is_winner(copy, player_letter):
                return i

    move = choose_random_move_from_list(board, [1, 3, 7, 9])
    if move is None:
        return (
            5
            if is_space_free(board, 5)
            else choose_random_move_from_list(board, [2, 4, 6, 8])
        )

    else:
        return move


def is_board_full(board):
    return not any(is_space_free(board, i) for i in range(1, 10))


print("Welcome To Tic Tac Toe!")

while True:
    the_board = [" "] * 10
    player_letter, computer_letter = input_player_letter()
    turn = frist_player()
    print(f"The {turn} go frist.")
    game_is_playing = True

    while game_is_playing:
        if turn == "player":
            # players turn
            draw_board(the_board)
            move = get_player_move(the_board)
            make_move(the_board, player_letter, move)

            if is_winner(the_board, player_letter):
                draw_board(the_board)
                print("Hoory! You have won the game!")
                game_is_playing = False
            elif is_board_full(the_board):
                draw_board(the_board)
                print("The game is tie!")
                break
            else:
                turn = "computer"
        else:
            # Computer's turn
            move = get_computer_move(the_board, computer_letter)
            make_move(the_board, computer_letter, move)

            if is_winner(the_board, computer_letter):
                draw_board(the_board)
                print("The computer has beaten you! You Lose.")
                game_is_playing = False
            elif is_board_full(the_board):
                draw_board(the_board)
                print("The game is a tie!")
                break
            else:
                turn = "player"
    if not play_again():
        break
