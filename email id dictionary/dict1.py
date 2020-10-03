#!/usr/bin/env python

counts=dict()
mails=list()
fname=input('Enter file name:')
fh=open(fname)
for line in fh:
    if not line.startswith('From'):
        continue
    if line.startswith('From:'):
        continue
    id=line.split()
    mail=id[1]
    mails.append(mail)
for x in mails:
    counts[x]=counts.get(x,0)+1
bigmail=None
bigvalue=None
for key,value in counts.items():
    if bigvalue==None or bigvalue<value:
        bigmail=key
        bigvalue=value
print(bigmail, bigvalue)
