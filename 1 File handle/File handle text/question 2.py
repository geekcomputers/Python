""" Write a method/function DISPLAYWORDS() in python to read lines
 from a text file STORY.TXT,
 using read function
and display those words, which are less than 4 characters. """


print("Hey!! You can print the word which are less then 4 characters")        

def display_words(file_path):


    try:
        with open(file_path, 'r') as F:
            lines = F.read()
            words = lines.split()
            count = 0
            for word in words:
                if (len(word) < 4):
                    print(word)
                    count += 1
        return "The total number of the word's count which has less than 4 characters", (count) 
    
    except FileNotFoundError:
        print("File not found")

print("Just need to pass the path of your file..")

file_path = input("Please, Enter file path: ")

if __name__ == "__main__":
    
    print(display_words(file_path))
                




