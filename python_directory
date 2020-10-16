n= int(input())
phoneBook = {}


for i in range(0,n):
    name, num = input().split()
    phoneBook[name]= num

def query(phoneBook, q):
    
    if q in phoneBook:
        num1 = phoneBook[q]
       
        print('%s=%s' % (q,num1))
    else:
        print("Not found")
         

inputs = []
try:
 while True:
    q = input()
    query(phoneBook,q)
except EOFError as e:
    print(end="")
