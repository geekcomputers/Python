__author__ = 'tusharsappal'
##  This script basically fetches  the file from the user specified location and finds the frequency of each word in the text file
def word_frequency_counter():
    fname=raw_input("Enter the Name of the file : ")
    try:
        fhand = open(fname)
    except:
        print("The file cannot  be opened",file)
        exit()

    counts= dict()
    for line in fhand:
        words = line.split()
        for word in words:
            if word not in counts:
                counts[word]=1
            else:
                counts[word]=counts[word]+1


    print counts

word_frequency_counter()