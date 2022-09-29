text = input("Enter Text : ")

for chr in text:
    bin = ""
    asciiVal = ord(chr)
    while asciiVal > 0:
        bin = f"{bin}0" if asciiVal % 2 == 0 else f"{bin}1"
        asciiVal //= 2
    print(f"{bin} : {bin[::-1]}")
