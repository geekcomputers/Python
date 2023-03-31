
import os
import time
file_name= input("Enter the file name to create:- ")

print(file_name)

def write_to_file(file_name):

    if os.path.exists(file_name):
        print(f"Error: {file_name} already exists.")

    else:
        with open(file_name, "a") as F:
            while True:
                text = input("enter any text to add in the file:- ")
                F.write(
                    text + "\n"
                )  # write function takes exactly 1 arguement so concatenation
                choice = input("Do you want to enter more, y/n")
                if choice == "n":
                    break
        
def longlines():
    with open(file_name, encoding='utf-8') as F:
        lines = F.readlines()

        for i in lines:
            if len(i) < 50:
                print(i, end="\t")
            else: 
                print("There is no line which is less than 50 ")


if __name__ == "__main__":
    write_to_file(file_name)
    time.sleep(1)
    longlines()