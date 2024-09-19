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

def upper_half_repeat_pattern(lines=5): 
     for column in range(1, (lines +1)): 
         print(f"{str(column) * column}") 


def lower_half_repeat_pattern(lines=5):
     for length in range(lines, 0, -1): 
         print(f"{str(length) * length}")


def upper_half_incremental_pattern(lines=5):
     const=""
     for column in range(1, (lines +1)):
         const+=str(column)
         print(const)



def lower_half_incremental_pattern(lines=5):
     for row_length in range(lines, 0, -1):
         for x in range(1,row_length+1):
             print(x,end='') 
         print()



if __name__ == "__main__":
    main()
