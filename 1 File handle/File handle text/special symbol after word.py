F = open("happy.txt", "r")
# method 1
val = F.read()
val = val.split()
for i in val:
    print(i, "*", end="")
print("\n")


# method 2
F.seek(0)
value = F.readlines()
for line in value:
    for word in line.split():
        print(word, "*", end="")
F.close()
