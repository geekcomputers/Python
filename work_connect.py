"""Connect or disconnect work resources (RDP, Checkpoint, PuTTYCM).

Usage:
  -c <password>: Connect using Checkpoint, launch PuTTYCM and RDP
  -d: Disconnect Checkpoint session
  -h/--help: Show this help text
"""

import os
import subprocess
import sys
import time

# Environment variable and file paths
dropbox: str | None = os.getenv("dropbox")
rdpfile: str = "remote\\workpc.rdp"
conffilename: str = os.path.join(dropbox, rdpfile) if dropbox else ""
remote: str = r"c:\windows\system32\mstsc.exe"

# Validate argument count
if len(sys.argv) < 2:
    print(__doc__.strip())
    sys.exit(1)

# Handle help flags
if any(arg in sys.argv for arg in ("-h", "--help")):
    print(__doc__.strip())
    sys.exit(0)

# Process connection/disconnection
if sys.argv[1].lower().startswith("-c"):
    if len(sys.argv) < 3:
        print("Error: -c requires a password argument\n")
        print(__doc__.strip())
        sys.exit(1)
    passwd: str = sys.argv[2]
    # Launch Checkpoint connection
    subprocess.Popen(
        r"c:\Program Files\Checkpoint\Endpoint Connect\trac.exe "
        f"connect -u username -p {passwd}"
    )
    # Launch PuTTY configuration manager
    subprocess.Popen(r"c:\geektools\puttycm.exe")
    # Wait for VPN to establish
    time.sleep(15)
    # Launch RDP session
    if conffilename:
        subprocess.Popen([remote, conffilename])
elif sys.argv[1].lower().startswith("-d"):
    # Disconnect Checkpoint session
    subprocess.Popen(
        r"c:\Program Files\Checkpoint\Endpoint Connect\trac.exe disconnect"
    )
else:
    print(f"Unknown option: {sys.argv[1]}\n")
    print(__doc__.strip())
    sys.exit(1)
