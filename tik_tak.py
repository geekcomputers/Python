
l=["anything",1,2,3,4,5,6,7,8,9]
i=0
j=9
print("\n\t\t\tTIK-TAC-TOE")
def board():
    #import os
    #os.system('cls')
    print("\n\n")
    print("    |     |" )
    print("",l[1]," | ",l[2]," | ",l[3] )
    print("____|_____|____")
    print("    |     |" )
    print("",l[4]," | ",l[5]," | ",l[6] )
    print("____|_____|____")
    print("    |     |" )
    print("",l[7]," | ",l[8]," | ",l[9] )
    print("    |     |" )
def enter_number(p1,p2):
    global i
    global j
    k=9
    while(j):
        if k==0:
            break
        
        if i==0:
            x=int(input("\nplayer 1 :- "))
            if x<=0:
                print("chose number from given board")
            else:
                for e in range(1,10):
                    if l[e]==x:
                        l[e]=p1
                        board()
                        c=checkwin()
                        if c==1:
                            print("\n\n Congratulation ! player 1 win ")
                            return
                        
                        
                        i=1
                        j-=1
                        k-=1
                        if k==0:
                            print("\n\nGame is over")
                            break
                        
        if k==0:
            
            break
                   
        if i==1:
            y=int(input("\nplayer 2 :- "))
            if y<=0:
                print("chose number from given board")
                #return
            else:
                for e in range(1,10):
                    if l[e]==y:
                        l[e]=p2
                        board()
                        w=checkwin()
                        if w==1:
                            print("\n\n Congratulation ! player 2 win")
                            return
                        
                        i=0
                        j-=1
                        k-=1
                        
                    
def checkwin():
    if l[1]==l[2]==l[3]:
        
        return 1
    elif l[4]==l[5]==l[6]:
        
        return 1
    elif l[7]==l[8]==l[9]:
        
        return 1
    elif l[1]==l[4]==l[7]:
        
        return 1

    elif l[2]==l[5]==l[8]:
        
        return 1
    elif l[3]==l[6]==l[9]:
        
        return 1
    elif l[1]==l[5]==l[9]:
        
        return 1
    elif l[3]==l[5]==l[7]:
        
        return 1
    else:
        print("\n\nGame continue")
        
def main():
    board()
    p1=input("\n\nplayer 1 chose your sign [0/x] = ")
    p2=input("player 2 chose your sign [0/x] = ")
    enter_number(p1,p2)
    print("\n\n\t\t\tDeveloped By :- UTKARSH MATHUR")
main()
