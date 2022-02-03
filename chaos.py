# A simple program illustrating chaotic behaviour


def main():
    print("This program illustrates a chaotic function")

    while True:
        try:
            x = float((input("Enter a number between 0 and 1: ")))
            if 0 < x and x < 1:
                break
            else:
                print("Please enter correct number")
        except Exception as e:
            print("Please enter correct number")

    for i in range(10):
        x = 3.9 * x * (1 - x)
        print(x)


if __name__ == "__main__":
    main()
