#Note :- Client and Server Must be connected to same Network
# import socket  modules
import socket

# create TCP/IP socket
s = socket.socket()
# take user input ip of server
server = input("Enter Server IP: ")
# bind the socket to the port 12345, and connect  
s.connect((server, 12345))
# receive message from server connection successfully established
data = s.recv(1024).decode("utf-8")
print(server + ": " + data)

while True:
    # send message to server
    new_data = str(input("You: ")).encode("utf-8")
    s.sendall(new_data)
    # receive message from server
    data = s.recv(1024).decode("utf-8")
    print(server + ": " + data)

# close connection
s.close()
