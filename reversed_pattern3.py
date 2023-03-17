#Simple inverted number triangle piramid
#11111
#2222
#333
#44
#5

def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    t = 1
    for i in reversed(range(1, (lines +1))):
        format = str(t)*i
        print(format)
        t = t + 1

if __name__ == "__main__":
    main()
