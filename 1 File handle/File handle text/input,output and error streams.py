#practicing with streams
import sys
sys.stdout.write("Enter the name of the file")
file=sys.stdin.readline()

F=open(file.strip(),"r")
    
while True:
    ch=F.readlines()
    for i in ch():        #ch is the whole file,for i in ch gives lines, for j in i gives letters,for j in i.split gives words
        print(i,end="") 
    else:
        sys.stderr.write("End of file reached")
        break
F.close()        
