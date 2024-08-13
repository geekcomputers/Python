#pattern Reverse piramid of numbers
#1
#21
#321
#4321
#54321

def main():
    lines = int(input("Enter the number of lines: "))
    pattern(lines)

def pattern(rows): 
    const=''
    for i in range(1, rows+1): 
        const=str(i)+const
        print(const)

if __name__ == "__main__":
    main()
