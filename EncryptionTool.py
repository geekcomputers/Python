#GGearing
#Simple encryption script for text
#This was one my first versions of this script
#09/07/2017

text=input("Enter text: ")
PI=3.14159265358979323846264338327950288419716939937510
text=list(text)
values= list()
reverse=list()
def encryptChar(target):
    #encrytion algorithm
    target=(((target+42)*PI)-449)
    return target

def decryptChar(target):
    target=(((target+449)/PI)-42)
    return target

def encrypt(input_text):
    input_text_list=list(input_text)
    col_values=list()
    for i in range (len(input_text_list)):
        current=ord(input_text_list[i])
        current=encryptChar(current)
        col_values.append(current)
    return col_values

def decrypt(enc_text):
    enc_list
    for i in range (len(input_text_list)):
        current=int(decryptChar(values[i]))
        current=chr(current)
        col_values.append(current)
    return col_values

def readAndDecrypt(filename):
    file=open(filename,"r")
    data=file.read()
    datalist=list()
    datalistint=list()
    actualdata=list()
    datalist=data.split(" ")
    datalist.remove('')
    for i in range(len(datalist)):
        datalistint.append(float(datalist[i]))
    for i in range(len(datalist)):
        current1=int(decryptChar(datalistint[i]))
        current1=chr(current1)
        actualdata.append(current1)
    return actualdata

def readAndEncrypt(filename):
    file=open(filename,"r")
    data=file.read()
    datalist=list(data)
    encrypted_list=list()
    encrypted_list_str=list()
    for i in range(len(datalist)):
        current=ord(datalist[i])
        current=encryptChar(current)
        encrypted_list.append(current)
    return encrypted_list

def readAndEncryptAndSave(inp_file,out_file):
    enc_list=readAndEncrypt(inp_file)
    output=open(out_file,"w")
    for i in range(len(enc_list)):
        output.write(str(enc_list[i])+" ")
    output.close()

def readAndDecryptAndSave(inp_file,out_file):
    dec_list=readAndDecrypt(inp_file)
    output=open(out_file,"w")
    for i in range(len(dec_list)):
        output.write(str(dec_list[i]))
    output.close()
#encryption
for i in range (len(text)):
    current=ord(text[i])
    current=encryptChar(current)
    values.append(current)

#decryption
for i in range (len(text)):
    current=int(decryptChar(values[i]))
    current=chr(current)
    reverse.append(current)
print(reverse)

#saves encrypted in txt file
output=open("encrypted.txt","w")
for i in range(len(values)):
    output.write(str(values[i])+" ")
output.close()

#read and decrypts
print(readAndDecrypt("encrypted.txt"))

    
