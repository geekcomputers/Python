    # (upper half - repeat)
    #1
    #22
    #333

    # (upper half - incremental)
    #1
    #12
    #123
  
    # (lower half - incremental)
    #123
    #12
    #1

    # (lower half - repeat)
    #333
    #22
    #1

def main():
    lines = int(input("Enter no.of lines: "))
    pattern = input("i: increment or r:repeat pattern: ").lower()
    part = input("u: upper part or l: lower part: ").lower()

    match pattern:
        case "i":
            if part == "u":
                upper_half_incremental_pattern(lines)
            else:
                lower_half_incremental_pattern(lines)

        case "r":
            if part == "u":
                upper_half_repeat_pattern(lines)
            else:
                lower_half_repeat_pattern(lines)

        case _:
            print("Invalid input")
            exit(0)

def upper_half_repeat_pattern(lines):

    t = 1
    for column in range(1, (lines +1)):
        print(f"{str(t) * column}")
        t += 1

def upper_half_incremental_pattern(lines):

    for column in range(1, (lines +1)):
        row = ""
        for ii in range(1, column +1):
            row += str(ii)
        print(row)
            

def lower_half_incremental_pattern(lines):

    for row_length in range(lines, 0, -1):
        row = ""
        column = 1

        for _ in range(row_length):
            column = 0 if column == 10 else column
            row = f"{row}{column}"
            column += 1

        print(row)

def lower_half_repeat_pattern(lines):

    for row_length in range(lines, 0, -1):
        
        row = ""
        for _ in range(1, row_length+1):
            row += str(row_length)
        print(row)

if __name__ == "__main__":
    main()
