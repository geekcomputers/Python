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
    target = (((target + 42) * key) - 449)
    return target


def decryptChar(target):
    target = (((target + 449) / key) - 42)
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
    file = open(filename, "r")
    data = file.read()
    datalistint = []
    actualdata = []
    datalist = data.split(" ")
    datalist.remove('')
    datalistint = [float(data) for data in datalist]
    for data in datalist:
        current1 = int(decryptChar(data))
        current1 = chr(current1)
        actualdata.append(current1)
    file.close()
    return actualdata


def readAndEncrypt(filename):
    file = open(filename, "r")
    data = file.read()
    datalist = list(data)
    encrypted_list = list()
    encrypted_list_str = list()
    for data in datalist:
        current = ord(data)
        current = encryptChar(current)
        encrypted_list.append(current)
    file.close()
    return encrypted_list


def readAndEncryptAndSave(inp_file, out_file):
    enc_list = readAndEncrypt(inp_file)
    output = open(out_file, "w")
    for enc in enc_list:
        output.write(str(enc) + " ")
    output.close()


def readAndDecryptAndSave(inp_file, out_file):
    dec_list = readAndDecrypt(inp_file)
    output = open(out_file, "w")
    for dec in dec_list:
        output.write(str(dec))
    output.close()


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

# saves encrypted in txt file
output = open("encrypted.txt", "w")
for v in values:
    output.write(str(v) + " ")
output.close()

# read and decrypts
print(readAndDecrypt("encrypted.txt"))
