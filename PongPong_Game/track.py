#track.py
coverage_dict = {
    "pong/ball.py/BallObject/update.if": False,
    "pong/ball.py/BallObject/update.elif": False,
    "pong/ball.py/BallObject/update.elif2": False,
    "pong/ball.py/BallObject/update.else": False,
    "pong/paddle.py/Paddle/update.if1": False,
    "pong/paddle.py/Paddle/update.elif1": False,
    "pong/paddle.py/Paddle/update.if2": False,
    "pong/paddle.py/Paddle/update.elif2": False,
}

def update_coverage(key):
    if key in coverage_dict:
        coverage_dict[key] = True

def print_coverage():
    for key, value in coverage_dict.items():
        print(f"{key}: {'Covered' if value else 'Not Covered'}")

def write_coverage_to_file(filename):
    with open(filename, 'w') as file:
        for key, value in coverage_dict.items():
            file.write(f"{key}: {'Covered' if value else 'Not Covered'}\n")
