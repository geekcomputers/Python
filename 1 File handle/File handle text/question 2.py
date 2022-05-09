""" Write a method/function DISPLAYWORDS() in python to read lines
 from a text file STORY.TXT,
 using read function
and display those words, which are less than 4 characters. """

def display_words():
    with open("story.txt") as F:
        lines = F.read()
        words = lines.split()
        count = 0
        for word in words:
            if (len(word) < 4):
                print(word)
                count += 1
    return count 

if __name__ == "__main__":
    print(display_words())
                




