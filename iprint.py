from time import sleep

txt = input("")

ap = ""

for let in range(len(txt) - 1):
    ap += txt[let]
    print(ap, end="\r")
    sleep(0.1)

print(txt, end="")
