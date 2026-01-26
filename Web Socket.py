# Program to print a data & it's Metadata of online uploaded file using "socket".
import socket
from colorama import Fore # this module for Color the font 

# handling the exceptions
try:
    skt_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    skt_c.connect(("data.pr4e.org", 80))
    link = "GET http://data.pr4e.org/intro-short.txt HTTP/1.0\r\n\r\n".encode()
    skt_c.send(link)
except(Exception) as e:
    # this code runes on error in any connection 
    print(Fore.RED, e, Fore.RESET)

while True:
    data = skt_c.recv(512)
    if len(data) < 1:
        break
    print(data.decode())
skt_c.close()
