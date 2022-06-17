# practicing with streams
import sys

sys.stdout.write("Enter the name of the file")
file = sys.stdin.readline()

with open(file.strip(), ) as F:

    while True:
        ch = F.readlines()
        for i in ch:
            print(i, end="")
        sys.stderr.write("End of file reached")
        break

