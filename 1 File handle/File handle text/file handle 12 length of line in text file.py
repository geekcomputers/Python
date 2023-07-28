
import os
import time
file_name= input("Enter the file name to create:- ")

print(file_name)

def write_to_file(file_name):

    if os.path.exists(file_name):
        print(f"Error: {file_name} already exists.")
        return

    with open(file_name, "a") as F:

        while True:
            text = input("enter any text to add in the file:- ")
            F.write( f"{text}\n" )
            choice = input("Do you want to enter more, y/n").lower()
            if choice == "n":
                break
    
def longlines():

    with open(file_name, encoding='utf-8') as F:
        lines = F.readlines()
        lines_less_than_50 = list( filter(lambda line: len(line) < 50, lines ) )

        if not lines_less_than_50:
            print("There is no line which is less than 50")
        else:
            for i in lines_less_than_50:
                print(i, end="\t")

if __name__ == "__main__":
    write_to_file(file_name)
    time.sleep(1)
    longlines()