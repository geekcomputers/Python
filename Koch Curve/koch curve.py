# importing the libraries
# turtle standard graphics library for python


# function to create koch snowflake or koch curve
def snowflake(lengthSide, levels):
    if levels == 0:
        forward(lengthSide)
        return
    lengthSide /= 3.0
    snowflake(lengthSide, levels - 1)
    left(60)
    snowflake(lengthSide, levels - 1)
    right(120)
    snowflake(lengthSide, levels - 1)
    left(60)
    snowflake(lengthSide, levels - 1)


# main function
if __name__ == "__main__":
    speed(0)  # defining the speed of the turtle
    length = 300.0  #
    penup()  # Pull the pen up – no drawing when moving.
    # Move the turtle backward by distance, opposite to the direction the turtle is headed.
    # Do not change the turtle’s heading.
    backward(length / 2.0)
    pendown()
    for i in range(3):
        # Pull the pen down – drawing when moving.
        snowflake(length, 4)
        right(120)
    # To control the closing windows of the turtle
    mainloop()
