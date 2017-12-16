# Python-Assembler

This program is a simple assembler-like (intel-syntax) interpreter language. The program is written in python 2. 
To start the program you will type 

``` python assembler.py code.txt ```


After you type 'enter' the program will interpret the source-code in 'code.txt'
You can use many texfiles as input. These will be interpret one by one.

You find some examples in the directory 'examples'.

For instance

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

**You find a guide in GUIDE.md**

