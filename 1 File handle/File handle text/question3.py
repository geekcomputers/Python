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
                F.write(f"{text}\n") 

                if input("do you want to enter more, y/n").lower() == "n":
                    break
        
# step2:
def check_first_letter():
    with open(file_name) as F:
        lines = F.read().split()

        # store all starting letters from each line in one string after converting to lower case
        first_letters = "".join([line[0].lower() for line in lines])

        count_i = first_letters.count("i")
        count_m = first_letters.count("m")

        print(f"The total number of sentences starting with I or M are {count_i + count_m}")

if __name__ == "__main__":
    
    write_to_file(file_name)
    time.sleep(1)
    check_first_letter()
