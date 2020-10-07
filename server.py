import socket 
import threading 

format = 'utf-8'
header = 64
connected_msg = 'DISCONNECTED'

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())

server = socket.socket(socket.AF_INET , socket.SOCK_STREAM) 

ADDR = (SERVER , PORT)
server.bind(ADDR) # The particular socket that we have just created is now bind with the declared port
                  # All the different connections are going to hit that particular socket at the time 
                  # of connection

def start():
    server.listen()
    print(f"[SERVER-CHECK :: ] The server is running ")
    print(f"[SERVER-IP-INFO :: ] {SERVER} ")
    print("Enter 'q' to stop this process ")
    while True:
        conn , addr = server.accept()
        thread = threading.Thread(target=handle_client , args=(conn , addr))
        thread.start()
        print(f"[ACTIVE CONNECTION :: ] {threading.activeCount() - 1}")
        handle_client(conn , addr)
        if(input() == 'q'):
            break
    print("The server is stopped !! watch you again")



## handling multiple clients together at the same time using threading
## Each client will use a thread on server so no one will going to wait for any one 

def handle_client(conn , addr):
    print(f"[New Connection :: ] {addr} is connected with this server ")
    connected = True
    while connected:
            msg = conn.recv(1024).decode(format)
            if msg:
                if msg == connected_msg:
                    print(f"{addr} is disconnected from the chat group ")
                    connected = False
                    conn.close()
                
                print(f"{addr} => {msg}")


print("The server is going to start!! wait ")
start()