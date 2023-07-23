#PROJECT1
#CAESAR CIPHER DECODER 

#Author: InTruder
#Cloned from: https://github.com/InTruder-Sec/caesar-cipher


def main():
    print("[>] CAESAR CIPHER DECODER!!! \n")
    print("[1] Encrypt\n[2] Decrypt")
    try:
        func = int(input("Choose one of the above(example for encode enter 1): "))
    except:
        print("\n[>] Invalid input")
        exit()

    if func == 2:
        decode()
    else:
        if func == 1:
            encode()
        else:
            print("\n[>] Invalid input")
        exit()

def encode():
    text = input("Enter text to encode: ")
    key = input("Enter number of characters you want to shift: ")
    encoded_cipher = ""
    try:
        key = int(key)
    except:
        print("Only intigers between 0 to 25 are allowed. Try again :)")
        exit()
    if key > 25:
        print("Only intigers between 0 to 25 are allowed. Try again :)")
        exit()
    else:
        key = key
    text = text.upper()
    for char in text:
        ascii = ord(char)
        if ascii > 90:
            new_ascii = ascii
        else:
            if ascii < 65:
                new_ascii = ascii
            else:
                new_ascii = ascii + key
                if  new_ascii > 90:
                    new_ascii = new_ascii - 26
                else:
                    new_ascii = new_ascii
        encoded = chr(new_ascii)
        encoded_cipher = encoded_cipher + encoded
    print("Encoded text: " + encoded_cipher)



def decode():
    cipher = input("\n[>] Enter your cipher text: ")
    print("Posiblities of cipher text are: \n")
    cipher = cipher.lower()
    for i in range(1, 26):
        decoded = ""
        decoded_cipher = ""
        for char in cipher:
            ascii = ord(char)
            if ascii < 97:
                new_ascii = ascii
            else:
                if ascii > 122:
                    new_ascii = ascii
                else:
                    new_ascii = ascii - int(i)
                    if new_ascii < 97:
                      new_ascii = new_ascii + 26
                    else:
                        new_ascii = new_ascii
            decoded = chr(new_ascii)
            decoded_cipher = decoded_cipher + decoded
        print("\n" + decoded_cipher)


if __name__ == '__main__':
    main()
