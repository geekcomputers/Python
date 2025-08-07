# pattern
# @@@@@@@@    $
# @@@@@@@    $$
# @@@@@@    $$$
# @@@@@    $$$$
# @@@@    $$$$$
# @@@    $$$$$$
# @@    $$$$$$$
# @    $$$$$$$$


def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)


def pattern(lines):
    t = 1
    for i in range(lines, 0, -1):
        nxt_pattern = "$" * t
        pattern = "@" * (i)
        final_pattern = pattern + "    " + nxt_pattern
        print(final_pattern)
        t = t + 1


if __name__ == "__main__":
    main()
