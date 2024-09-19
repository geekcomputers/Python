#pattern
#$$$$$$$$$$$
# $$$$$$$$$
#  $$$$$$$
#   $$$$$
#    $$$
#     $



def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
     flag=lines
     for i in range(lines):
         print(" "*(i),'$'*(2*flag-1))
         flag-=1

if __name__ == "__main__":
    main()

