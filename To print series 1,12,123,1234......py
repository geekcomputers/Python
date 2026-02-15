def print_pattern(rows: int) -> None:
    for i in range(1, rows + 1):
        print("".join(str(j) for j in range(1, i + 1)))


def start():
    while True:
        try:
            n = int(input("Enter number of rows: "))
            if n < 1:
                print("Invalid value, enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input, please enter a number.")

    print_pattern(n)


start()
