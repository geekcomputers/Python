# Polyline drawing in codeskulptor

import simplegui

polyline = []


def click(pos):
    global polyline
    polyline.append(pos)


def clear():
    global polyline
    polyline = []


def draw(canvas):
    if len(polyline) == 1:
        canvas.draw_point(polyline[0], "White")
    for i in range(1, len(polyline)):
        canvas.draw_line(polyline[i - 1], polyline[i], 2, "White")


frame = simplegui.create_frame("Echo click", 300, 200)
frame.set_mouseclick_handler(click)
frame.set_draw_handler(draw)
frame.add_button("Clear", clear)

frame.start()
