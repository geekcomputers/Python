""" Write a method/function DISPLAYWORDS() in python to read lines
 from a text file STORY.TXT,
 using read function
and display those words, which are less than 4 characters. """


print("Hey!! You can print the word which are less then 4 characters")        

def display_words(file_path):

    try:
        with open(file_path) as F:
            words = F.read().split()
            words_less_than_40 = list( filter(lambda word: len(word) < 4, words) )

            for word in words_less_than_40:
                print(word)
        
        return "The total number of the word's count which has less than 4 characters", (len(words_less_than_40)) 
    
    except FileNotFoundError:
        print("File not found")

print("Just need to pass the path of your file..")

file_path = input("Please, Enter file path: ")

if __name__ == "__main__":
    
    print(display_words(file_path))
                




