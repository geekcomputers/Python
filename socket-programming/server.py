# Client and Server Must be connected to same network
# import socket module
import socket

# create TCP/IP socket
s = socket.socket()
# get the according IP address
ip = socket.gethostbyname(socket.gethostname())
# binding ip address and port
s.bind((ip, 12345))
# listen for incoming connections (server mode) with 3 connection at a time
s.listen(3)
# print your ip address
print("Server ip address:", ip)
while True:
    # waiting for a connection establishment
    print("waiting for a connection")
    connection, client_address = s.accept()
    try:
        # show connected client
        print("connected from", client_address)
        # sending acknowledgement to client that you are connected
        connection.send(str("Now You are connected").encode("utf-8"))

        # receiving the message
        while True:
            data = connection.recv(1024).decode("utf-8")
            if data:
                # message from client
                print(list(client_address)[0], end="")
                print(": %s" % data)
                # Enter your message to send to client
                new_data = str(input("You: ")).encode("utf-8")
                connection.send(new_data)
    finally:
        # Close connection
        connection.close()
