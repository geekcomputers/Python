#pattern Reverse piramid of numbers
#1
#21
#321
#4321
#54321

def main():
    pattern(int(input("Enter the number of lines: ")))

def pattern(rows):
    for i in range(1, rows+1):
        for j in range(i, 0, -1):
            print(j, end="")
        print()

if __name__ == "__main__":
    main()
