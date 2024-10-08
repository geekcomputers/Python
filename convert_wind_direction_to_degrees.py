def degrees_to_compass(degrees):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(degrees / 45) % 8
    return directions[index]

# Taking input from the user
while True:
    try:
        degrees = float(input("Enter the wind direction in degrees (0-359): "))
        if degrees < 0 or degrees >= 360:
            raise ValueError("Degrees must be between 0 and 359")
        break
    except ValueError as ve:
        print(f"Error: {ve}")
        continue


compass_direction = degrees_to_compass(degrees)
print(f"{degrees} degrees is {compass_direction}")
