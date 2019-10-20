#!/usr/bin/python
# coding: utf-8

import math
import os
import sys


def slice(mink, maxk):
    s = 0.0
    for k in range(int(mink), int(maxk)):
        s += 1.0 / (2 * k + 1) / (2 * k + 1)
    return s


def pi(n):
    pids = []
    unit = n / 10
    for i in range(10):  # 分10个子进程
        mink = unit * i
        maxk = mink + unit
        pid = os.fork()
        if pid > 0:
            pids.append(pid)
        else:
            s = slice(mink, maxk)  # 子进程开始计算
            with open("%d" % os.getpid(), "w") as f:
                f.write(str(s))
            sys.exit(0)  # 子进程结束
    sums = []
    for pid in pids:
        os.waitpid(pid, 0)  # 等待子进程结束
        with open("%d" % pid, "r") as f:
            sums.append(float(f.read()))
        os.remove("%d" % pid)  # 删除通信的文件
    return math.sqrt(sum(sums) * 8)


if __name__ == '__main__':
    print("start")
    print(pi(10000000))
