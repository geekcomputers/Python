# To Find The Total Number Of Digits In A Number

N = int(input("Enter The number"))
count = 0

while(N!=0):
    N = (N-N%10)/10
    count+=1


print(count)        