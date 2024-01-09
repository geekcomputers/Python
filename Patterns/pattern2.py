#pattern
#$$$$$$$$$$$
# $$$$$$$$$
#  $$$$$$$
#   $$$$$
#    $$$
#     $



def main():
    pattern(int(input("Enter no.of lines: ")))

def pattern(lines):
    for i in range(lines,0,-1):
        for j in range(lines-i):
            print(' ', end='') 
        
        for j in range(2*i-1):
            print('$',end='')
        print() 


if __name__ == "__main__":
    main()

