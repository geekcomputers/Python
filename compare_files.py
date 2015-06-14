## this script compares  the two text files
## first line of defence is to match the number of line in the file
## second line of defence is to store the content in two arrays and then compare
__author__ = 'tusharsappal'

def compare_two_files(str_1,str_2):
    print "Comparing starts"
    print"First fetching the number of line"
    with open(str_1) as fin :
        lines = sum (1 for line in fin)
    with open(str_2) as fin_2:
        lines_2 = sum(1 for line in fin_2)

    if(lines== lines_2):
        print "First Line of defence is passed"
        ins = open (str_1,"r")
        array = []
        for line_1 in ins:
            array.append(line_1)
        ins.close()
        ins_2 = open (str_2,"r")
        array_2=[]
        for line_2 in ins_2:
            array_2.append(line_2)
        ins_2.close()
        if (array==array_2):
            print "In the Second Line of Defence the files have passed"
        else :
            print "In the second line of defence the files have failed"
        
        
    else :
        print "First line of defence is not passed"



## replace the first parameter with the first file path and the second paramter with the second file path 
compare_two_files("path to the first file ","path to the second file")
        
    
