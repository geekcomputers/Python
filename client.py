import socket 
import threading

format = 'utf-8'

header = 64
server = '192.168.139.1'
port = 5050
connected_msg = 'DISCONNECTED'

ADDR = (server, port)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send_request():
    msg = input(f"{server} :: ")
    msg = msg.encode(format)
    client.send(msg)

send_request()