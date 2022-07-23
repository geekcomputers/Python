#pattern
#$$$$$$$$$$$
# $$$$$$$$$
#  $$$$$$$
#   $$$$$
#    $$$
#     $



def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    t = 0
    m = lines + 1
    for i in reversed(range(lines+1)):
        pattern = "@"*(m)
        format = " "*t
        t = t + 1
        print(format + pattern)
        m = m -2
        if m <= 0:
            exit()

if __name__ == "__main__":
    main()
