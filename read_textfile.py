# declaring a global array_store to store the data temporarily 
__author__ = 'tusharsappal'
array_store=[]
def readFile(str):
    print "reading starts from this part"
    f=open(str,"r")
    global array_store
    
    for line in f:
        line=line.rstrip('\n')
        array_store.append(line)
        
    for l in array_store:
        print l
    
    return ;


# enter the path in side the readFile method call statement
readFile("Enter the text file path separated  by / slashes ")




