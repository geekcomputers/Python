import socket
import threading

flag = 0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
hostname = input("Enter your host :: ")
s.connect((hostname, 1023))
nickname = input("Enter your Name :: ")


def recieve():
    while True:
        try:
            msg = s.recv(1024).decode("utf-8")
            if msg == "NICK":
                print("Welcome to Chat room :: ", nickname)
                s.send(bytes(nickname, "utf-8"))
            else:
                print(msg)
        except Exception as error:
            print(f"An Erro occured {error}")
            s.close()
            flag = 1
            break


def Write():
    while True:
        try:
            reply_msg = f"{nickname} :: {input()}"
            s.send(bytes(reply_msg, "utf-8"))
        except Exception as error:
            print(f"An Error Occured while sending message !!!\n error : {error}")
            s.close()
            flag = 1
            break


if flag == 1:
    exit()
recieve_thrd = threading.Thread(target=recieve)
recieve_thrd.start()

write_thrd = threading.Thread(target=Write)
write_thrd.start()
