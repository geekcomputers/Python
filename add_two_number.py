user_input = (input("type type 'start' to run program:")).lower()

if user_input == 'start':
    is_game_running = True
else:
    is_game_running = False


while (is_game_running):
    num1 = int(input("enter number 1:"))
    num2 = int(input("enter number 2:"))
    num3 = num1+num2
    print(f"sum of {num1} and {num2} is {num3}")
    user_input = (input("if you want to discontinue type 'stop':")).lower()
    if user_input == "stop":
        is_game_running = False
        
