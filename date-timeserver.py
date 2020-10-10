import socket
import time

soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
soc.bind((socket.gethostname(),2905))
soc.listen(5)
while True:
    
    clientsocket,addr = soc.accept()      

    print("estavlishes  a connection from %s" % str(addr))
    currentTime = time.ctime(time.time()) + "\r\n"
    clientsocket.send(currentTime.encode('ascii'))
    clientsocket.close()
