def is_armstrong_number(number):
    numberstr = str(number)
    length = len(numberstr)
    num = number
    rev = 0
    temp = 0
    while num != 0:
        rem = num % 10
        num //= 10
        temp += rem ** length
    return temp == number


if __name__ == "__main__":
    is_armstrong_number(int(input("Enter number: ")))
