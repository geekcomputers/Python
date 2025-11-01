with open("happy.txt", "r") as F:
    # method 1
    for i in F.read().split():
        print(i, "*", end="")
    print("\n")

    # method 2
    F.seek(0)
    for line in F.readlines():
        for word in line.split():
            print(word, "*", end="")
