from constants import *
from arena import * 

def over(screen, score):

    heading = "GAME OVER!"
    selection = 0
    menu_options = ['MAIN MENU', 'QUIT']

    while True:
        screen.clear()
        height, width = screen.getmaxyx()
        arena(screen, score)
        try:

            if height < HEIGHT or width < WIDTH:
                raise curses.error

            score_string = f"FINAL SCORE : {score}"
            screen.addstr(HEIGHT//3 - 6, (WIDTH // 2) - len(heading)//2  , heading)        
            screen.addstr(HEIGHT//3 - 3, (WIDTH // 2) - len(score_string)//2  , score_string)        
            screen.addstr(HEIGHT//3 + 6 + selection * 2, (WIDTH // 2) - 10, ">>")
            for i, option in enumerate(menu_options):
                screen.addstr(HEIGHT//3 + 6 + i * 2, (WIDTH // 2) - len(option)//2, option)

            screen.refresh() 
        except curses.error:
            return
        key = screen.getch()

        if key in (S, curses.KEY_DOWN):
            selection = selection + 1 if selection < 1 else selection
        elif key in (W, curses.KEY_UP):
            selection = selection - 1 if selection > 0 else selection
        elif key == ord('\n'):
            return menu_options[selection]
