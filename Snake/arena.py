import curses
from constants import *
from pause import pause

def arena(screen, score = 0):
    height, width = screen.getmaxyx()

    if height < HEIGHT or width < WIDTH:
        pause(screen, height, width)
        screen.refresh()
    else:
        try:
            for r in range(HEIGHT):
                if r == 0:
                    screen.addch(r, 0, curses.ACS_ULCORNER)
                    for c in range(1, WIDTH - 1):
                        screen.addch(r, c, curses.ACS_HLINE)
                    screen.addch(r, WIDTH - 1, curses.ACS_URCORNER)
                elif r == 1:
                    score_string = f"SCORE : {score}"
                    screen.addstr(r, 1, score_string)         
                    for c in (0, WIDTH-1):
                        screen.addch(r, c, curses.ACS_VLINE)
                elif r == HEIGHT - 1:
                    screen.addch(r, 0, curses.ACS_LLCORNER)
                    for c in range(1, WIDTH - 1):
                        screen.addch(r, c, curses.ACS_HLINE)
                    screen.addch(r, WIDTH - 1, curses.ACS_LRCORNER)
                elif r == 2:
                    screen.addch(r, 0, curses.ACS_LTEE)
                    for c in range(1, WIDTH - 1):
                        screen.addch(r, c, curses.ACS_HLINE)
                    screen.addch(r, WIDTH - 1, curses.ACS_RTEE)
                else:
                    screen.addch(r, 0, curses.ACS_VLINE)
                    for c in range(1, WIDTH - 1):
                        screen.addch(r, c, " ")
                    screen.addch(r, WIDTH - 1, curses.ACS_VLINE)
        except curses.error:
            pass
