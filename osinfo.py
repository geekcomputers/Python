# Script Name		: osinfo.py
# Author				: Craig Richards
# Created				: 5th April 2012
# Last Modified		: 
# Version				: 1.0

# Modifications		: 

# Description			: Displays some information about the OS you are running this script on

import platform

profile = [
platform.architecture(),
platform.dist(),
platform.libc_ver(),
platform.mac_ver(),
platform.machine(),
platform.node(),
platform.platform(),
platform.processor(),
platform.python_build(),
platform.python_compiler(),
platform.python_version(),
platform.release(),
platform.system(),
platform.uname(),
platform.version(),
]
i=1
for item in profile:
  print '#',i,' ',item
  i=i+1;
