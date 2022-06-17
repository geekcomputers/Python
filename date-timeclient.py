import socket

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect((socket.gethostname(), 2905))
recmsg = soc.recv(1024)
soc.close()
print(f'The time got from the server is {recmsg.decode("ascii")}')
