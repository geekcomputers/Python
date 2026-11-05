import curses
from time import sleep
from menu import * 
from arena import arena
from pause import *
from game import *
from over import *
from sys import exit

def main(screen):
    screen.keypad(True)
    curses.noecho()
    curses.curs_set(0)
    try:
        while True:
            screen.clear()
            
            height, width = screen.getmaxyx()

            if height < HEIGHT or width < WIDTH:
                pause(screen, height, width)
            else:  
                arena(screen)
                screen.refresh()
                setting = menu(screen)
                score = game(screen, setting)
                selection = over(screen, score)

                if selection == "MAIN MENU":
                    continue
                else:
                    exit(0)

            screen.refresh()
            key = screen.getch()
            if key == ord('q'):
                break
    finally:
        screen.keypad(False)

if __name__ == "__main__":
    curses.wrapper(main)
