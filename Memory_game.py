import random

import simplegui


def new_game():
    global card3, po, state, exposed, card1

    def create(card):
        while len(card) != 8:
            num = random.randrange(0, 8)
            if num not in card:
                card.append(num)
        return card

    card3 = []
    card1 = []
    card2 = []
    po = []
    card1 = create(card1)
    card2 = create(card2)
    card1.extend(card2)
    random.shuffle(card1)
    state = 0
    exposed = []
    for i in range(0, 16, 1):
        exposed.insert(i, False)


def mouseclick(pos):
    global card3, po, state, exposed, card1
    if state == 2:
        if card3[0] != card3[1]:
            exposed[po[0]] = False
            exposed[po[1]] = False
        card3 = []
        state = 0
        po = []
    ind = pos[0] // 50
    card3.append(card1[ind])
    po.append(ind)
    if exposed[ind] == False and state < 2:
        exposed[ind] = True
        state += 1


def draw(canvas):
    global card1
    gap = 0
    for i in range(0, 16, 1):
        if exposed[i] == False:
            canvas.draw_polygon(
                [[0 + gap, 0], [0 + gap, 100], [50 + gap, 100], [50 + gap, 0]],
                1,
                "Black",
                "Green",
            )
        elif exposed[i] == True:
            canvas.draw_text(str(card1[i]), [15 + gap, 65], 50, "White")
        gap += 50


frame = simplegui.create_frame("Memory", 800, 100)
frame.add_button("Reset", new_game)
label = frame.add_label("Turns = 0")

frame.set_mouseclick_handler(mouseclick)
frame.set_draw_handler(draw)

new_game()
frame.start()
