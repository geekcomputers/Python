# Script Name	: new_script.py
# Author			: Craig Richards
# Created			: 20th November 2012
# Last Modified	: 
# Version			: 1.0

# Modifications	: 

# Description		: This will create a new basic template for a new script

import os				# Load the library module
import sys				# Load the library module
import datetime		# Load the library module

text = '''You need to pass an argument for the new script you want to create, followed by the script name.  You can use
	-python	: Python Script
	-bash	: Bash Script
	-ksh	: Korn Shell Script
	-sql	: SQL Script'''

if len(sys.argv) < 3:
  print text				
  sys.exit()			
  
if '-h' in sys.argv or '--h' in sys.argv or '-help' in sys.argv or '--help' in sys.argv:	
  print text																									
  sys.exit()																								
else:	
  if '-python' in sys.argv[1]:
    config_file="python.cfg"
    extension=".py"
  elif '-bash' in sys.argv[1]:
    config_file="bash.cfg"
    extension=".bash"
  elif '-ksh' in sys.argv[1]:
    config_file="ksh.cfg"
    extension=".ksh"
  elif '-sql' in sys.argv[1]:
    config_file="sql.cfg"
    extension=".sql"
  else:
    print 'Unknown option - ' + text						
    sys.exit()

confdir=os.getenv("my_config")
scripts=os.getenv("scripts")
dev_dir="Development"
newfile=sys.argv[2]
output_file=(newfile+extension)
outputdir=os.path.join(scripts,dev_dir)
script=os.path.join(outputdir, output_file)
input_file=os.path.join(confdir,config_file)
old_text=" Script Name	: "
new_text=(" Script Name	: "+output_file)
if not(os.path.exists(outputdir)):
  os.mkdir(outputdir)
newscript = open(script, 'w')								
input=open(input_file,'r')
today=datetime.date.today()
old_date= " Created	:"
new_date= (" Created	: "+today.strftime("%d %B %Y"))
	
for line in input:			
  line = line.replace(old_text, new_text)
  line = line.replace(old_date, new_date)
  newscript.write(line)				
