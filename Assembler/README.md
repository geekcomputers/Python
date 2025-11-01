# hy your name 
# Python-Assembler
# WE need A FREE T-SHIRT
This program is a simple assembler-like (intel-syntax) interpreter language. The program is written in python 3. 
To start the program you will need to type 

``` python assembler.py code.txt ```


After you hit 'enter' the program will interpret the source-code in 'code.txt'.
You can use many textfiles as input. These will be interpreted one by one.

You can find some examples in the directory 'examples'.

For instance-

``` 
$msg db "hello world"

mov ecx, $msg 
mov eax, 4
int 0x80
mov eax, 1
mov ebx, 0
int 0x80
``` 

Will print onto console

```
hello world
END PROGRAM
```

**Refer to GUIDE.txt to read a guide**
