def degree_to_direction(deg):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    deg = deg % 360
    deg = int(deg // 45)
    print(directions[deg])


degree_to_direction(45)
