#Simple number triangle piramid
#1
#22
#333
#4444
#55555
#666666




def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    t = 1
    for i in range(1, (lines +1)):
        format = str(t)*i
        print(format)
        t = t + 1

if __name__ == "__main__":
    main()
