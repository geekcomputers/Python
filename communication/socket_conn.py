# coding: utf-8

from __future__ import print_function

import math
import os
import socket
import sys


def slice(mink, maxk):
    s = 0.0
    for k in range(mink, maxk):
        s += 1.0 / (2 * k + 1) / (2 * k + 1)
    return s


def pi(n):
    childs = {}
    unit = n / 10
    for i in range(10):  # 分10个子进程
        mink = unit * i
        maxk = mink + unit
        rsock, wsock = socket.socketpair()
        pid = os.fork()
        if pid > 0:
            childs[pid] = rsock
            wsock.close()
        else:
            rsock.close()
            s = slice(mink, maxk)  # 子进程开始计算
            wsock.send(str(s))
            wsock.close()
            sys.exit(0)  # 子进程结束
    sums = []
    for pid, rsock in childs.items():
        sums.append(float(rsock.recv(1024)))
        rsock.close()
        os.waitpid(pid, 0)  # 等待子进程结束
    return math.sqrt(sum(sums) * 8)


print(pi(10000000))
