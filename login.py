import os
from getpass import getpass

# Devloped By Black_angel
# This is Logo Function
def logo():
    print(" ──────────────────────────────────────────────────────── ")
    print(" |                                                        | ")
    print(" |   ########    ##  #########  ##       ##      ###      | ")
    print(" |   ##     ##   ##  ##         ##       ##    ##   ##    | ")
    print(" |   ##     ###  ##  ##         ##       ##   ##     ##   | ")
    print(" |   ##     ###  ##  #########  ###########  ##########   | ")
    print(" |   ##     ###  ##         ##  ##       ##  ##      ##   | ")
    print(" |   ##     ##   ##         ##  ##       ##  ##      ##   | ")
    print(" |   ########    ##  #########  ##       ##  ##      ##   | ")
    print(" |                                                        | ")
    print(" \033[1;91m|   || Digital Information Security Helper Assistant ||  | ")
    print(" |                                                        | ")
    print(" ──────────────────────────────────────────────────────── ")
    print("\033[1;36;49m")


# This is Login Function
def login():
    # for clear the screen
    os.system("clear")
    print("\033[1;36;49m")
    logo()
    print("\033[1;36;49m")
    print("")
    usr = input("Enter your Username : ")
    # This is username you can change here
    usr1 = "raj"
    psw = getpass("Enter Your Password : ")
    # This is Password you can change here
    psw1 = "5898"
    if usr == usr1 and psw == psw1:
        print("\033[1;92mlogin successfully")
        os.system("clear")
        print("\033[1;36;49m")
        logo()
    else:
        print("\033[1;91m Wrong")

        login()


# This is main function
if __name__ == "__main__":
    login()
