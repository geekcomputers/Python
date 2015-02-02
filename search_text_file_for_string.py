## This script searches a particular string in the text file
__author__ = 'tusharsappal'
def read_file(str,string_to_be_searched):
    print "The reading starts here"
    with open(str,"r") as input_data:
        for line in input_data:
            if line.find(string_to_be_searched)>-1 :
                print "String found"
                break



## replace the first argument with the file path separated with forward slashes and second argument with string to be searched
read_file("path_to_the_text_file","string_to_be_searched")
            
    
       
