# PROJECT1
# CAESAR CIPHER ENCODER/DECODER

# Author: InTruder
# Cloned from: https://github.com/InTruder-Sec/caesar-cipher

# Improved by: OfficialAhmed (https://github.com/OfficialAhmed)

def get_int() -> int:
    """
    Get integer, otherwise redo
    """

    try:
        key = int(input("Enter number of characters you want to shift: "))
    except:
        print("Enter an integer")
        key = get_int()

    return key

def main():

    print("[>] CAESAR CIPHER DECODER!!! \n")
    print("[1] Encrypt\n[2] Decrypt")

    match input("Choose one of the above(example for encode enter 1): "):

        case "1":
            encode()

        case "2":
            decode()

        case _:
            print("\n[>] Invalid input. Choose 1 or 2")
            main()


def encode():

    encoded_cipher = ""
    text = input("Enter text to encode: ")
    key = get_int()
        
    for char in text:
        
        ascii = ord(char) + key
        encoded_cipher += chr(ascii)

    print(f"Encoded text: {encoded_cipher}")


def decode():

    decoded_cipher = ""
    cipher = input("\n[>] Enter your cipher text: ")
    key = get_int()

    for character in cipher:
        ascii = ord(character) - key
        decoded_cipher += chr(ascii)

    print(decoded_cipher)


if __name__ == '__main__':
    main()
