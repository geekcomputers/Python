#pattern Reverse piramid of numbers
#1
#21
#321
#4321
#54321

def main():
    lines = int(input("Enter number of lines: "))
    pattern(lines)

def pattern(rows):


    for row in range(1, rows):
        for column in range(row, 0, -1):
            print(column, end=' ')
        print("")




if __name__ == "__main__":
    main()