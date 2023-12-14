'''' Multiple extension changer'''
import time
from pathlib import Path as p
import random as rand
import hashlib


def chxten_(files, xten):
    chfile = []
    for file in files:
        ch_file = file.split('.')
        ch_file = ch_file[0]
        chfile.append(ch_file)
    if len(xten) == len(chfile):
        chxten = []
        for i in range(len(chfile)):
            ch_xten = chfile[i] + xten[i]
            chxten.append(ch_xten)
    elif len(xten) < len(chfile) and len(xten) != 1:
        chxten = []
        for i in range(len(xten)):
            ch_xten = chfile[i] + xten[i]
            chxten.append(ch_xten)
        for i in range(1, (len(chfile) + 1) - len(xten)):
            ch_xten = chfile[- + i] + xten[-1]
            chxten.append(ch_xten)
    elif len(xten) == 1:
        chxten = []
        for i in range(len(chfile)):
            ch_xten = chfile[i] + xten[0]
            chxten.append(ch_xten)
    elif len(xten) > len(chfile):
        chxten = []
        for i in range(1, (len(xten) + 1) - len(chfile)):
            f = p(files[-i])
            p.touch(chfile[-i] + xten[-1])
            new = f.read_bytes()
            p(chfile[-i] + xten[-1]).write_bytes(new)
        for i in range(len(chfile)):
            ch_xten = chfile[i] + xten[i]
            chxten.append(ch_xten)
    else:
        return 'an error occured'
    return chxten


# End of function definitions
# Beggining of execution of code
#password
password = input('Enter password:')

password = password.encode()

password = hashlib.sha512(password).hexdigest()
if password == 'c99d3d8f321ff63c2f4aaec6f96f8df740efa2dc5f98fccdbbb503627fd69a9084073574ee4df2b888f9fe2ed90e29002c318be476bb62dabf8386a607db06c4':
    pass
else:
    print('wrong password!')
    time.sleep(0.3)
    exit(404)
files = input('Enter file names and thier extensions (seperated by commas):')
xten = input('Enter Xtensions to change with (seperated by commas):')

if files == '*':
    pw = p.cwd()
    files = ''
    for i in pw.iterdir():
        if not p.is_dir(i):
            i = str(i)
            if not i.endswith('.py'):
                # if not i.endswith('exe'):
                if not i.endswith('.log'):
                    files = files + i + ','
if files == 'r':
    pw = p.cwd()
    files = ''
    filer = []
    for i in pw.iterdir():
        if p.is_file(i):
            i = str(i)
            if not i.endswith('.py'):
                if not i.endswith('.exe'):
                    if not i.endswith('.log'):
                        filer.append(i)
    for i in range(5):
        pos = rand.randint(0,len(filer))
        files = files + filer[pos] + ','

    print(files)
files = files.split(',')
xten = xten.split(',')

# Validation
for file in files:
    check = p(file).exists()
    if check == False:
        print(f'{file} is not found. Paste this file in the directory of {file}')
        files.remove(file)
# Ended validation

count = len(files)
chxten = chxten_(files, xten)

# Error Handlings
if chxten == 'an error occured':
    print('Check your inputs correctly')
    time.sleep(1)
    exit(404)
else:
    try:
        for i in range(len(files)):
            f = p(files[i])
            f.rename(chxten[i])
        print('All files has been changed')
    except PermissionError:
        pass
    except FileNotFoundError:
        # Validation
        for file in files:
            check = p(file).exists()
            if check == False:
                print(f'{file} is not found. Paste this file in the directory of {file}')
                files.remove(file)
    # except Exception:
    #     print('An Error Has Occured in exception')
    #     time.sleep(1)
    #     exit(404)

# last modified 3:25PM 12/12/2023 (DD/MM/YYYY)
