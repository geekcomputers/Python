"""
This is the main file that runs the Snake game.
It handles screen setup, dynamic boundaries, UI controls (buttons),
game state management, and the main game loop.
"""
from turtle import Screen, Turtle
from snake import Snake
from food import Food
from scoreboard import Scoreboard
from wall import Wall
import colors

# --- CONSTANTS ---
MOVE_DELAY_MS = 100  # Game speed in milliseconds

# --- GAME STATE ---
game_state = "start"  # Possible states: "start", "playing", "paused", "game_over"

# --- SCREEN SETUP ---
screen = Screen()
screen.setup(width=0.9, height=0.9)  # Set up a nearly fullscreen window
screen.bgcolor(colors.BG_COLOR)
screen.title("Interactive Snake Game")
screen.tracer(0)

# --- DYNAMIC GAME BOUNDARIES ---
WIDTH = screen.window_width()
HEIGHT = screen.window_height()
# These boundaries are calculated to be inside the visible wall with a safe margin
LEFT_WALL = -WIDTH / 2 + 25
RIGHT_WALL = WIDTH / 2 - 25
TOP_WALL = HEIGHT / 2 - 85
BOTTOM_WALL = -HEIGHT / 2 + 25

# --- GAME OBJECTS ---
wall = Wall()
snake = Snake()
food = Food()
# Initial food placement is now handled after boundaries are calculated
food.refresh(LEFT_WALL, RIGHT_WALL, BOTTOM_WALL, TOP_WALL)
scoreboard = Scoreboard()

# --- UI CONTROLS (BUTTONS) ---
buttons = {}  # Dictionary to hold button turtles and their properties

def create_button(name, x, y, width=120, height=40):
    """Creates a turtle-based button with a label."""
    if name in buttons and buttons[name]['turtle'] is not None:
        buttons[name]['turtle'].clear()

    button_turtle = Turtle()
    button_turtle.hideturtle()
    button_turtle.penup()
    button_turtle.speed("fastest")

    button_turtle.goto(x - width/2, y - height/2)
    button_turtle.color(colors.BUTTON_BORDER_COLOR, colors.BUTTON_BG_COLOR)
    button_turtle.begin_fill()
    for _ in range(2):
        button_turtle.forward(width)
        button_turtle.left(90)
        button_turtle.forward(height)
        button_turtle.left(90)
    button_turtle.end_fill()

    button_turtle.goto(x, y - 12)
    button_turtle.color(colors.BUTTON_TEXT_COLOR)
    button_turtle.write(name, align="center", font=("Lucida Sans", 14, "bold"))

    buttons[name] = {'turtle': button_turtle, 'x': x, 'y': y, 'w': width, 'h': height, 'visible': True}

def hide_button(name):
    """Hides a button by clearing its turtle."""
    if name in buttons and buttons[name]['visible']:
        buttons[name]['turtle'].clear()
        buttons[name]['visible'] = False

def manage_buttons():
    """Shows or hides buttons based on the current game state."""
    all_buttons = ["Play", "Pause", "Resume", "Restart"]
    for btn_name in all_buttons:
        hide_button(btn_name)

    btn_x = WIDTH / 2 - 100
    btn_y = HEIGHT / 2 - 45

    if game_state == "start":
        create_button("Play", 0, -100)
    elif game_state == "playing":
        create_button("Pause", btn_x, btn_y)
    elif game_state == "paused":
        create_button("Resume", btn_x, btn_y)
    elif game_state == "game_over":
        create_button("Restart", btn_x, btn_y)

# --- GAME LOGIC & STATE TRANSITIONS ---
def start_game():
    global game_state
    if game_state == "start":
        game_state = "playing"
        scoreboard.update_scoreboard()

def toggle_pause_resume():
    global game_state
    if game_state == "playing":
        game_state = "paused"
        scoreboard.display_pause()
    elif game_state == "paused":
        game_state = "playing"
        scoreboard.update_scoreboard()

def restart_game():
    global game_state
    if game_state == "game_over":
        game_state = "playing"
        snake.reset()
        food.refresh(LEFT_WALL, RIGHT_WALL, BOTTOM_WALL, TOP_WALL)
        scoreboard.reset()

def is_click_on_button(name, x, y):
    """Checks if a click (x, y) is within the bounds of a visible button."""
    if name in buttons and buttons[name]['visible']:
        btn = buttons[name]
        return (btn['x'] - btn['w']/2 < x < btn['x'] + btn['w']/2 and
                btn['y'] - btn['h']/2 < y < btn['y'] + btn['h']/2)
    return False

def handle_click(x, y):
    """Main click handler to delegate actions based on button clicks."""
    if game_state == "start" and is_click_on_button("Play", x, y):
        start_game()
    elif game_state == "playing" and is_click_on_button("Pause", x, y):
        toggle_pause_resume()
    elif game_state == "paused" and is_click_on_button("Resume", x, y):
        toggle_pause_resume()
    elif game_state == "game_over" and is_click_on_button("Restart", x, y):
        restart_game()

# --- KEYBOARD HANDLERS ---
def handle_snake_up():
    if game_state in ["start", "playing"]:
        start_game()
        snake.up()
def handle_snake_down():
    if game_state in ["start", "playing"]:
        start_game()
        snake.down()
def handle_snake_left():
    if game_state in ["start", "playing"]:
        start_game()
        snake.left()
def handle_snake_right():
    if game_state in ["start", "playing"]:
        start_game()
        snake.right()

# --- KEY & MOUSE BINDINGS ---
screen.listen()
screen.onkey(handle_snake_up, "Up")
screen.onkey(handle_snake_down, "Down")
screen.onkey(handle_snake_left, "Left")
screen.onkey(handle_snake_right, "Right")
screen.onkey(toggle_pause_resume, "space")
screen.onkey(restart_game, "r")
screen.onkey(restart_game, "R")
screen.onclick(handle_click)

# --- MAIN GAME LOOP ---
def game_loop():
    global game_state
    if game_state == "playing":
        snake.move()
        # Collision with food
        if snake.head.distance(food) < 20:
            food.refresh(LEFT_WALL, RIGHT_WALL, BOTTOM_WALL, TOP_WALL)
            snake.extend()
            scoreboard.increase_score()
        # Collision with wall
        if not (LEFT_WALL < snake.head.xcor() < RIGHT_WALL and BOTTOM_WALL < snake.head.ycor() < TOP_WALL):
            game_state = "game_over"
            scoreboard.game_over()
        # Collision with tail
        for segment in snake.segments[1:]:
            if snake.head.distance(segment) < 10:
                game_state = "game_over"
                scoreboard.game_over()
    manage_buttons()
    screen.update()
    screen.ontimer(game_loop, MOVE_DELAY_MS)

# --- INITIALIZE GAME ---
scoreboard.display_start_message()
game_loop()
screen.exitonclick()

