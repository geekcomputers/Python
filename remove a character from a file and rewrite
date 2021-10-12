#Remove all the lines that contain the character `a' in a file and write it to another file.
f=open("test1.txt","r") #opening file test1.txt
lines = f.readlines() #saved lines
print("Original file is :")
print(lines)
f.close()
 
# Rewriting lines 

e=open("test3.txt","w") # file containing lines with 'a'
f=open("test1.txt","w") # file containing lines without 'a'
for line in lines:
 if 'a' in line or 'A' in line:
  e.write(line)
 else:        
  f.write(line)
    
e.close()
f.close()   

f=open("test1.txt","r")   
lines=f.readlines()

e=open("test3.txt","r")   
lines1=e.readlines()

print("\n")

print("Files without letter a:")
print(lines)
print("\n")

print("Files with letter a:")
print(lines1)

e.close()
f.close()
