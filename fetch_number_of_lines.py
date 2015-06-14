## fetching the number of lines from a text file

__author__ = 'tusharsappal'
def fetch_number_of_lines(str):
    print "Fetching starts from this point"
    with open(str) as fin:
        lines = sum (1 for line in fin)
        print "Number of lines is "
        print lines





## replace the argument in the function call  to the path of the file

fetch_number_of_lines("The path to the text file")

        
    

