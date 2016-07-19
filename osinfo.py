# Script Name		: osinfo.py
# Authors		: {'geekcomputers': 'Craig Richards', 'dmahugh': 'Doug Mahugh','rutvik1010':'Rutvik Narayana Nadimpally','y12uc231': 'Satyapriya Krishna', 'minto4644':'Mohit Kumar'}
# Created		: 5th April 2012
# Last Modified	        : July 19 2016
# Version		: 1.0

# Modification 1	: Changed the profile to list again. Order is important. Everytime we run script we don't want to see different ordering.
# Modification 2        : Fixed the AttributeError checking for all properties. Using hasttr(). 
# Modification 3        : Removed ': ' from properties inside profile.  


# Description		: Displays some information about the OS you are running this script on

import platform as pl

profile = [
'architecture',
'linux_distribution',
'mac_ver',
'machine',
'node',
'platform',
'processor',
'python_build',
'python_compiler',
'python_version',
'release',
'system',
'uname',
'version',
]



for key in profile:
    if hasattr(pl,key):
        print(key + ": "+ str(getattr(pl,key)()))
        
