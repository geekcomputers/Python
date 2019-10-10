# Author- RIZWAN AHMAD 

# Simple python Script to change mac address of linux generate random or enter mac address


import random
from subprocess import PIPE, Popen


# function for returning terminal command
def cret(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]


# function for genrate mac address random
def randmac():
    return [0x00, 0x16, 0x3e,
            random.randint(0x00, 0x7f),
            random.randint(0x00, 0xff),
            random.randint(0x00, 0xff)]


def retrandmac(mac):
    return ':'.join(map(lambda x: "%02x" % x, mac))


print("                                             +-+-+-+ +-+-+-+-+-+-+-+")
print("                                             |M|A|C| |c|h|a|n|g|e|r|")
print("                                             +-+-+-+ +-+-+-+-+-+-+-+")
# finding wireless interface name that should start with wl e.g.-wlan0,wlp3s0
infname = cret('ifconfig -a  | egrep "^[wl-wl]+" | sed "s/: .*//" | grep -v "lo"')
# INTERFACE NAME 6 character so return 6 last character
infname = infname[:6]
infname = infname.decode('utf-8')
# GETTING MAC Address from /sys/class/net/wlan0/address directory
cmdgetmac = ('cat /sys/class/net/' + infname + '/address')
crrntmac = cret("cat /sys/class/net/" + infname + "/address")
crrntmac = crrntmac.decode('utf-8')
print(
    "Your Current mac address = " + crrntmac + "\nEnter Option to change Your MAC:\n1. Enter MAC address manually \n2. Automatic Random MAC address")
opt = int(input())

if opt == 1:
    print("Please Enter Your New MAC address: \nExmple:  46:d2:f4:0c:2a:50")

    newmac = input()
    print("Please wait changing  mac address..................")

    # first turn off wifi
    cret('nmcli radio wifi off')

    changemaccmd = "sudo ip link set dev " + infname + " address " + newmac
    # executing command with new mac address
    cret(changemaccmd)
    # turning on wifi
    cret('nmcli radio wifi on')
    # GETTING MAC Address from /sys/class/net/wlan0/address directory
    cr = cret("cat /sys/class/net/" + infname + "/address")
    cr = cr.decode('utf-8')

    print("\nNow Your Current mac address = " + cr)


elif opt == 2:
    genmac = retrandmac(randmac())
    print("Please wait generating new mac address.....................")
    cret('nmcli radio wifi off')
    changemaccmd = "sudo ip link set dev " + infname + " address " + genmac
    cret(changemaccmd)
    cret('nmcli radio wifi on')
    cr = cret("cat /sys/class/net/" + infname + "/address")
    cr = cr.decode('utf-8')
    print("Now Your Current mac address = " + cr)

else:
    print("You Have Selected wrong Option")
