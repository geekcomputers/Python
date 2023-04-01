"""Write a function in python to count the number of lowercase
alphabets present in a text file “happy.txt"""

import time
import os

print("You will see the count of lowercase, uppercase and total count of alphabets in provided file..")


file_path = input("Please, Enter file path: ")

if os.path.exists(file_path):
    print('The file exists and this is the path:\n',file_path) 


def lowercase(file_path):
    try:

        with open(file_path, 'r') as F:
            # Define the initial count of the lower and upper case.
            lowercase_count = 0
            uppercase_count = 0

            value = F.read()

            for i in value:
                if i.islower():
                    # It will increase the count.
                    lowercase_count += 1
                elif i.isupper():
                    uppercase_count += 1
            


            total_count = lowercase_count+uppercase_count
            
            print("The total number of lower case letters are", lowercase_count)
            time.sleep(1)
            print("The total number of upper case letters are", uppercase_count)
            time.sleep(1)
            print("The total number of letters are", total_count)
            time.sleep(1)

    except FileNotFoundError:
        print("File is not exist.. Please check AGAIN")




if __name__ == "__main__":

    lowercase(file_path)
