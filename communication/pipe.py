# coding: utf-8

from __future__ import print_function

import math
import os
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
        r, w = os.pipe()
        pid = os.fork()
        if pid > 0:
            childs[pid] = r  # 将子进程的pid和读描述符存起来
            os.close(w)  # 父进程关闭写描述符，只读
        else:
            os.close(r)  # 子进程关闭读描述符，只写
            s = slice(mink, maxk)  # 子进程开始计算
            os.write(w, str(s))
            os.close(w)  # 写完了，关闭写描述符
            sys.exit(0)  # 子进程结束
    sums = []

    for pid, r in childs.items():
        sums.append(float(os.read(r, 1024)))
        os.close(r)  # 读完了，关闭读描述符
        os.waitpid(pid, 0)  # 等待子进程结束
    return math.sqrt(sum(sums) * 8)


print(pi(10000000))
