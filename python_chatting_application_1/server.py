import socket
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1023))
print(socket.gethostname())
s.listen(5)

clients = []
nickename = []


def Client_Handler(cli):
    while True:
        try:
            reply = cli.recv(1024).decode("utf-8")
            if reply == "QUIT":
                index_of_cli = clients.index(cli)
                nick = nickename[index_of_cli]
                nickename.remove(nick)
                clients.remove(cli)
                BroadCasating(f"{nick} has left the chat room")
                print(f"Disconnected with f{nick}")
                break
            BroadCasating(reply)
        except Exception:
            index_of_cli = clients.index(cli)
            print(index_of_cli)
            nick = nickename[index_of_cli]
            nickename.remove(nick)
            clients.remove(cli)
            BroadCasating(f"{nick} has left the chat room")
            print(f"Disconnected with {nick}")
            break


def BroadCasating(msg):
    for client in clients:
        client.send(bytes(msg, "utf-8"))


def recieve():
    while True:
        client_sckt, addr = s.accept()
        print(f"Connection has been established {addr}")
        client_sckt.send(bytes("NICK", "utf-8"))
        nick = client_sckt.recv(1024).decode("utf-8")
        nickename.append(nick)
        clients.append(client_sckt)
        print(f"{nick} has joined the chat room ")
        BroadCasating(f"{nick} has joined the chat room say hi !!!")
        threading._start_new_thread(Client_Handler, (client_sckt,))


recieve()
s.close()
