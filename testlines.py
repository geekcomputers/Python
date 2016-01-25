# Script Name		: testlines.py
# Author				: Craig Richards
# Created				: 08th December 2011
# Last Modified		: 
# Version				: 1.0

# Modifications		: beven nyamande

# Description			: This very simple script open a file and prints out 100 lines of whatever is set for the line variableest you want to print\n"	# This sets the variable for the text that you want to print


def write_to_file(filename,txt):
  with open(filename,'w') as file_object:
      s = file_object.write(txt)
      
    
if __name__ == '__main__':
    write_to_file('test.txt', 'i am beven')

