def collatz_steps(n):
    times = 0
    while n != 1:
        if n % 2 == 0:
            print(f"{n} / 2", end=" ")
            n = n // 2 # "//" is a floor division where it rounds down the result
        else:
            print(f"{n} * 3 + 1", end=" ")
            n = 3 * n + 1
        print(f"= {n}")
        times += 1
    print(f"The number of times to reach 1 is {times}")

def main():
    again = "y"
    while again != "n":
        n = int(input("Input a number: "))
        collatz_steps(n)
        while True:
            again = str(input("Want to input again? y/n: "))
            if again != "n" and again != "y":
                print("Incorrect Input.")
            elif again == "n":
                print("Thank You! Goodbye.")
                break
            else: 
                break

main()
