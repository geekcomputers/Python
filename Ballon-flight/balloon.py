import pgzrun
from random import randint

# Screen size
WIDTH = 800
HEIGHT = 600

# Balloon
balloon = Actor("balloon")
balloon.pos = 400, 300

# Obstacles
bird = Actor("bird-up")
bird.pos = randint(800, 1600), randint(10, 200)

house = Actor("house")
house.pos = randint(800, 1600), 460

tree = Actor("tree")
tree.pos = randint(800, 1600), 450

# Restart button (appears when game over)
restart_btn = Rect((350, 350), (100, 40))

# Game variables
bird_up = True
up = False
game_over = False
score = 0
number_of_updates = 0
scores = []


# Step 8: Update High Scores
def update_high_scores():
    global score, scores
    filename = "high-scores.txt"
    scores = []

    with open(filename, "r") as file:
        line = file.readline()
        high_scores = line.split()
        for high_score in high_scores:
            if score > int(high_score):
                scores.append(str(score) + " ")
                score = int(high_score)
            else:
                scores.append(str(high_score) + " ")

    with open(filename, "w") as file:
        for high_score in scores:
            file.write(high_score)


# Step 9: Display High Scores
def display_high_scores():
    screen.draw.text("HIGH SCORES", (340, 150), color="black")
    y = 175
    position = 1
    for high_score in scores:
        screen.draw.text(
            f"Position {position}: {high_score}",
            (350, y),
            color="black",
        )
        y += 25
        position += 1

    # Restart button
    screen.draw.filled_rect(restart_btn, "lightblue")
    screen.draw.text("Restart", (restart_btn.x + 15,
                     restart_btn.y + 10), color="black")


# Step 10: Draw function
def draw():
    screen.blit("background", (0, 0))
    if not game_over:
        balloon.draw()
        bird.draw()
        house.draw()
        tree.draw()
        screen.draw.text(f"Score: {score}", (700, 5))
    else:
        display_high_scores()


# Step 11: Mouse control
def on_mouse_down(pos):
    global up
    if not game_over:
        up = True
        balloon.y -= 50
    else:
        # Check if restart clicked
        if restart_btn.collidepoint(pos):
            restart_game()


def on_mouse_up():
    global up
    up = False


# Step 12: Bird flapping
def flap():
    global bird_up
    if bird_up:
        bird.image = "bird-down"
        bird_up = False
    else:
        bird.image = "bird-up"
        bird_up = True


# Step 13: Update loop
def update():
    global game_over, score, number_of_updates

    if not game_over:
        # Balloon gravity
        if not up:
            balloon.y += 1

        # Bird movement
        if bird.x > 0:
            bird.x -= 4
            if number_of_updates == 9:
                flap()
                number_of_updates = 0
            else:
                number_of_updates += 1
        else:
            bird.x = randint(800, 1600)
            bird.y = randint(10, 200)
            score += 1
            number_of_updates = 0

        # House movement
        if house.right > 0:
            house.x -= 2
        else:
            house.x = randint(800, 1600)
            score += 1

        # Tree movement
        if tree.right > 0:
            tree.x -= 2
        else:
            tree.x = randint(800, 1600)
            score += 1

        # Check for collision or out of bounds
        if balloon.top < 0 or balloon.bottom > 560:
            game_over = True
            update_high_scores()

        if (
            balloon.collidepoint(bird.x, bird.y)
            or balloon.collidepoint(house.x, house.y)
            or balloon.collidepoint(tree.x, tree.y)
        ):
            game_over = True
            update_high_scores()


# Restart Game Function
def restart_game():
    global game_over, score, up, balloon, bird, house, tree, number_of_updates

    score = 0
    up = False
    game_over = False
    number_of_updates = 0

    balloon.pos = (400, 300)
    bird.pos = (randint(800, 1600), randint(10, 200))
    house.pos = (randint(800, 1600), 460)
    tree.pos = (randint(800, 1600), 450)


pgzrun.go()
