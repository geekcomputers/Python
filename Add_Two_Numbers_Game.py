user_input = (input("Type 'start' to run program:")).lower()

if user_input == "start":
    is_game_running = True
else:
    is_game_running = False


while is_game_running:
    num1 = int(input("Enter number 1:"))
    num2 = int(input("Enter number 2:"))
    num3 = num1 + num2
    print(f"The sum of {num1} and {num2} is {num3}")
    user_input = (input("If you want to end the game, type 'stop':")).lower()
    if user_input == "stop":
        is_game_running = False
