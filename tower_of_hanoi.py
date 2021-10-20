'''Tower of Hanoi is a mathematical puzzle where we have three rods and n disks. The objective of the puzzle is to move
    the entire stack to another rod, obeying the following simple rules:
1) Only one disk can be moved at a time.
2) Each move consists of taking the upper disk from one of the stacks and placing it on top of another stack i.e. a disk
    can only be moved if it is the uppermost disk on a stack.
3) No disk may be placed on top of a smaller disk.
APPROACH:
Take an example for 2 disks :
Let rod 1 = 'SOURCE', rod 2 = 'TEMPORARY', rod 3 = 'DESTINATION'.

Step 1 : Shift first disk from 'SOURCE' to 'TEMPORARY'.
Step 2 : Shift second disk from 'SOURCE' to 'DESTINATION'.
Step 3 : Shift first disk from 'TEMPORARY' to 'DESTINATION'.

The pattern here is :
Shift 'n-1' disks from 'SOURCE' to 'TEMPORARY'.
Shift last disk from 'SOURCE' to 'DESTINATION'.
Shift 'n-1' disks from 'TEMPORARY' to 'DESTINATION'.
'''
def toh(n,s,t,d):
    if n==1:
        print(s,'-->',d)
        return
    toh(n-1,s,d,t)
    print(s,'-->',d)
    toh(n-1,t,s,d)

if __name__=="__main__":
    while 1:

        n = int(input('''Enter number of disks:'''))

        if n<0:
            print("Try Again with a valid input")
            continue
        elif n==0:
            break
        toh(n,'Source','Temporary','Destination')

        print('ENTER 0 TO EXIT')

