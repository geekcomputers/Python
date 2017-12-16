# Guide for Python-Assembler

### Register

In this programming language you can use four registers. 
* eax
* ebx
* ecx
* edx

The register **eax** will be standard use for multiplication and division. 
Commands for arithmetic are:

* add  p0, p1
* sub  p0, p1
* mul  [register]
* div   [register]

p0 and p1 stands for parameter. p0 must be a register, p1 can be a register or constant.
The commands **mul** and **div** standard use eax. For instance:

```
mov ecx, 56
sub ecx, 10
mov eax, 4
int 0x80 

```

* The first line move the number  56 into register ecx.
* The second line subtract 10 from the ecx register.
* The third line move the number 4 into the eax register. This is for the print-function. 
* The fourt line call interrupt 0x80, thus the result will print onto console.
* The fifth line is a new line. This is important.

**Important: close each line with a newline!**

### System-Functions

With the interrupt 0x80 you can use some functionality in your program. 

EAX  | Function 
---- | ---------
1   | exit program. error code in ebx
3   | read input. onto ecx (only float)
4   | output onto console. print content in ecx

EAX stands for the register eax

### Variables

Variables begin with a $ or written in uppercase.

For instance:

```

; variables
VAR1 db 56
$var1 db 10

mov ecx, VAR1
mov ebx, $var1
sub ecx, ebx
mov eax, 4
int 0x80

```

**Important: The arithmetic commands (add, sub) works only with registers or constans. 
Therefore we must use the register ebx as a placeholder, above.**


Result of code, above.

```
46
```

### Comments

Comments begin with ; and ends with a newline. 
We noticed a comment, above. 

### Push and Pop

Sometimes we must save the content of a register, against losing of data.
Therefor we use the push and pop command.

```
push eax

```

This line will push the content of register eax onto the stack. 

```
pop ecx 

```

This line will pop the content of the top of the stack onto the ecx register. 

```
push [register]
pop [register]

```

### Jumps

With the command **cmp** we can compare two register. 

```
cmp r0, r1
je l1
jmp l2

```

Are the two register equal? The the command **je** is actively and jumps to label **l1**
Otherwise the command **jmp** is actively and jumps to label **l2**

#### Labels

For instance

```
l1: 

```

is a label.
Labels begin with a **l** and contains numbers.
For instance l1, l2 etc ...

To set a label you must end with a colon.
If you use a label in the jump commands, then avoid the colon at the end.

### Subprograms

```
mov ecx, 5

call _double
call _cube
call _inc

mov eax, 4
int 0x80
mov eax, 1
mov ebx, 0
int 0x80



_double:
add ecx, ecx
ret 

_cube:
push eax
mov eax, ecx
add ecx, eax
add ecx, eax
pop eax
ret

_inc:
add ecx, 1
ret

```

A subprogram label begins with a **_** and ends with a colon. See above. 


If you call the subprogram you must avoid the colon.

``` call _subprogramName
```

**Important:** Each subprogram must end with the **ret** command.