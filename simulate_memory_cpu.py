#! /user/bin/env python
# -*- encoding: utf-8 -*-
"""
Simulate cpu、 memory usage
"""

import sys
import re
import time
from multiprocessing import Process, cpu_count


def print_help():
    print('Usage: ')
    print('  python cpu_memory_simulator.py m 1GB')
    print('  python cpu_memory_simulator.py c 1')
    print('  python cpu_memory_simulator.py mc 1GB 2')

# memory usage


def mem():
    pattern = re.compile('^(\d*)([M|G]B)$')
    size = sys.argv[2].upper()
    match = pattern.match(size)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == 'MB':
            s = ' ' * (num * 1024 * 1024)
        else:
            s = ' ' * (num * 1024 * 1024 * 1024)
        time.sleep(24 * 3600)
    else:
        print("bad args.....")
        print_help()

# cpu usage


def deadloop():
    while True:
        pass

# Specify how many cores to occupy according to the parameters


def cpu():
    arg = sys.argv[2] if len(sys.argv) == 3 else sys.argv[3]
    cpu_num = cpu_count()
    cores = int(arg)
    if not isinstance(cores, int):
        print("bad args not int")
        return

    if cores > cpu_num:
        print("Invalid CPU Num(cpu_count="+str(cpu_num)+")")
        return

    if cores is None or cores < 1:
        cores = 1

    for i in range(cores):
        Process(target=deadloop).start()


def mem_cpu():
    Process(target=mem).start()
    Process(target=cpu).start()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        switcher = {
            'm': mem,
            'c': cpu,
            'mc': mem_cpu
        }
        switcher.get(sys.argv[1], mem)()
    else:
        print_help()
