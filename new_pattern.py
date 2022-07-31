#pattern
#@@@@@@@@    $
#@@@@@@@    $$
#@@@@@@    $$$
#@@@@@    $$$$
#@@@@    $$$$$
#@@@    $$$$$$
#@@    $$$$$$$
#@    $$$$$$$$

def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    t = 1
    for i in reversed(range(lines)):
        nxt_pattern = "$"*t
        pattern = "@"*(i+1)
        final_pattern = pattern + "   n " + nxt_pattern
        print(final_pattern)
        t = t +1

if __name__ == "__main__":
    main()