"""Write a user-defined function named count() that will read
the contents of text file named “happy.txt” and count
the number of lines which starts with either “I‟ or “M‟."""

import os
import time
file_name= input("Enter the file name to create:- ")

# step1:
print(file_name)



def write_to_file(file_name):

    if os.path.exists(file_name):
        print(f"Error: {file_name} already exists.")

    else:
        with open(file_name, "a") as F:
            while True:
                text = input("enter any text")
                F.write(
                    text + "\n"
                )  # write function takes exactly 1 arguement so concatenation
                choice = input("do you want to enter more, y/n")
                if choice == "n":
                    break
        
# write_to_file()

# step2:
def check_first_letter():
    with open(file_name) as F:
        value = F.read()
        count = 0
        line = value.split()
        for i in line:
            if i[0] in ["m", "M", "i", "I"]:
                count += 1
                print(i)
        print("The total number of sentences starting with I or M are", count)

if __name__ == "__main__":
    
    write_to_file(file_name)
    time.sleep(1)
    check_first_letter()
