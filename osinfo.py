# Script Name		: osinfo.py
# Author				: Craig Richards
# Created				: 5th April 2012
# Last Modified	: April 02 2016
# Version				: 1.0

# Modifications		: Changed the list to a dictionary. Although the order is lost, the info is with its label.

# Description			: Displays some information about the OS you are running this script on

import platform

profile = {
'Architecture: ': platform.architecture(),
#'Linux Distribution: ': platform.linux_distribution(),
'mac_ver: ': platform.mac_ver(),
'machine: ': platform.machine(),
'node: ': platform.node(),
'platform: ': platform.platform(),
'processor: ': platform.processor(),
'python build: ': platform.python_build(),
'python compiler: ': platform.python_compiler(),
'python version: ': platform.python_version(),
'release: ': platform.release(),
'system: ': platform.system(),
'uname: ': platform.uname(),
'version: ': platform.version(),
}

if hasattr(platform, 'linux_distribution'): 
    #to avoid AttributeError exception in some old versions of the module	
    profile['linux_distribution'] = platform.linux_distribution()
    #FIXME: do this for all properties but in a loop

for key in profile:
    print(key + str(profile[key]))
