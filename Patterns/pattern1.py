#pattern
#1234567
#123456
#12345
#1234
#123
#1

def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    m = lines + 1
    l = 1
    for i in reversed(range(lines+1)):
        t = ""
        k = 1
        for m in range(i):
            if k == 10:
                k = 0
            t = str(t) + str(k)
            k = k + 1
        print(t)

if __name__ == "__main__":
    main()
