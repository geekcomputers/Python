n=int(input("Enter number :"))
i=1
c=0
while(i<=n):
    if(n%i==0):
        c=c+1
    i=i+1
if(c<=2):
    print(n,"is prime number")
else:
    print(n,"is not a prime number")
        
