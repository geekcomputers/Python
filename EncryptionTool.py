# GGearing
# Simple encryption script for text
# This was one my first versions of this script
# 09/07/2017
from __future__ import print_function

import math

try:
    input = raw_input
except NameError:
    pass

key = int(math.pi * 1e14)
text = input("Enter text: ")
values = reverse = []


def encryptChar(target):
    # encrytion algorithm
    target = ((target + 42) * key) - 449
    return target


def decryptChar(target):
    target = ((target + 449) / key) - 42
    return target


def encrypt(input_text):
    col_values = []
    for inp in input_text:
        current = ord(inp)
        current = encryptChar(current)
        col_values.append(current)
    return col_values


def decrypt(enc_text):
    col_values = []
    for enc in enc_text:
        current = int(decryptChar(enc))
        current = chr(current)
        col_values.append(current)
    return col_values


def readAndDecrypt(filename):
    with open(filename, "r") as file:
        data = file.read()
        datalistint = []
        actualdata = []
        datalist = data.split(" ")
        datalist.remove("")
        datalistint = [float(data) for data in datalist]
        for data in datalist:
            current1 = int(decryptChar(data))
            current1 = chr(current1)
            actualdata.append(current1)
    return actualdata


def readAndEncrypt(filename):
    with open(filename, "r") as file:
        data = file.read()
        datalist = list(data)
        encrypted_list = []
        encrypted_list_str = []
        for data in datalist:
            current = ord(data)
            current = encryptChar(current)
            encrypted_list.append(current)
    return encrypted_list


def readAndEncryptAndSave(inp_file, out_file):
    enc_list = readAndEncrypt(inp_file)
    with open(out_file, "w") as output:
        for enc in enc_list:
            output.write(f"{str(enc)} ")


def readAndDecryptAndSave(inp_file, out_file):
    dec_list = readAndDecrypt(inp_file)
    with open(out_file, "w") as output:
        for dec in dec_list:
            output.write(str(dec))


# encryption
for t in text:
    current = ord(t)
    current = encryptChar(current)
    values.append(current)

# decryption
for v in values:
    current = int(decryptChar(v))
    current = chr(current)
    reverse.append(current)
print(reverse)

with open("encrypted.txt", "w") as output:
    for v in values:
        output.write(f"{str(v)} ")
# read and decrypts
print(readAndDecrypt("encrypted.txt"))
