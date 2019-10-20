from __future__ import print_function

import signal
import threading
from time import sleep


class producer(threading.Thread):
    def __init__(self, event):
        threading.Thread.__init__(self)
        self.event = event

    def run(self):
        while self.event.is_set():
            print("sub thread")
            sleep(2)
        else:
            print("sub thread end")
            exit()


def handler_thread(event):
    print("main thread end")
    event.clear()


def handler(signum, frame):
    handler_thread(frame.f_globals['event'])


signal.signal(signal.SIGINT, handler)

print("main thread")
event = threading.Event()
event.set()
p = producer(event)
p.start()
p.join()

sleep(100)  # 一定要使主线程处于活动状态，否则信号处理对子线程不起作用
