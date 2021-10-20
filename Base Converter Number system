def base_check(xnumber, xbase):
    for char in xnumber[len(xnumber ) -1]:
        if int(char) >= int(xbase):
            return False
    return True

def convert_from_10(xnumber, xbase, arr, ybase):
    if int(xbase) == 2 or int(xbase) == 4 or int(xbase) == 6 or int(xbase) == 8:

        if xnumber == 0:
            return arr
        else:
            quotient = int(xnumber) // int(xbase)
            remainder = int(xnumber) % int(xbase)
            arr.append(remainder)
            dividend = quotient
            convert_from_10(dividend, xbase, arr, base)
    elif int(xbase) == 16:
        if int(xnumber) == 0:
            return arr
        else:
            quotient = int(xnumber) // int(xbase)
            remainder = int(xnumber) % int(xbase)
            if remainder > 9:
                if remainder == 10: remainder = 'A'
                if remainder == 11: remainder = 'B'
                if remainder == 12: remainder = 'C'
                if remainder == 13: remainder = 'D'
                if remainder == 14: remainder = 'E'
                if remainder == 15: remainder = 'F'
            arr.append(remainder)
            dividend = quotient
            convert_from_10(dividend, xbase, arr, ybase)
def convert_to_10(xnumber, xbase, arr, ybase):
    if int(xbase) == 10:
        for char in xnumber:
            arr.append(char)
        flipped = arr[::-1]
        ans = 0
        j = 0

        for i in flipped:
            ans = ans + (int(i) * (int(ybase) ** j))
            j = j + 1
        return ans
arrayfrom = []
arrayto = []
is_base_possible = False
number = input("Enter the number you would like to convert: ")

while not is_base_possible:
    base = input("What is the base of this number? ")
    is_base_possible = base_check(number, base)
    if not is_base_possible:
        print(f"The number {number} is not a base {base} number")
        base = input
    else:
        break
dBase = input("What is the base you would like to convert to? ")
if int(base) == 10:
    convert_from_10(number, dBase, arrayfrom, base)
    answer = arrayfrom[::-1]  # reverses the array
    print(f"In base {dBase} this number is: ")
    print(*answer, sep='')
elif int(dBase) == 10:
    answer = convert_to_10(number, dBase, arrayto, base)
    print(f"In base {dBase} this number is: {answer} ")
else:
    number = convert_to_10(number, 10, arrayto, base)
    convert_from_10(number, dBase, arrayfrom, base)
    answer = arrayfrom[::-1]
    print(f"In base {dBase} this number is: ")
    print(*answer, sep='')
© 2020 GitHub, Inc.
