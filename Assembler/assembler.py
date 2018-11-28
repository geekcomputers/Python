import sys

lines = [] # contains the lines of the file. 
tokens = [] # contains all tokens of the source code.

#register eax, ebx,..., ecx
eax = 0
ebx = 0
ecx = 0
edx = 0

# status register
zeroFlag = False

# stack data structure
# push --> append
# pop --> pop
stack = []

# jump link table 
jumps = {}

# variable table
variables = {}

# return stack for subprograms
returnStack = []


# simple exception class 
class InvalidSyntax (Exception):
    def __init__ (self):
        pass

# class for represent a token
class Token():
    def __init__ (self, token, t):
        self.token = token
        self.t = t


# def initRegister():
#     global register
#     for i in range(9):
#         register.append(0)

def loadFile(fileName):
    """
        loadFile: This function loads the file and reads its lines.
    """
    global lines
    fo = open(fileName)
    for line in fo:
        lines.append(line)
    fo.close()


def scanner(string):
    """
        scanner: This function builds the tokens by the content of the file.
        The tokens will be saved in list 'tokens'
    """
    global tokens
    token = ""
    state = 0 # init state

    for ch in string:

        if state == 0:

            if ch == 'm':   # catch mov-command

                state = 1
                token += 'm'

            elif ch == 'e': # catch register

                state = 4
                token += 'e'

            elif (ch >= '1' and ch <= '9') or ch == '-':    # catch a number

                state = 6
                token += ch

            elif ch == '0': # catch a number or hex-code

                state = 17
                token += ch

            elif ch == 'a': # catch add-command

                state = 7
                token += ch

            elif ch == 's': # catch sub command

                state = 10
                token += ch

            elif ch == 'i': # capture int command

                state = 14
                token += ch

            elif ch == 'p': # capture push or pop command

                state = 19
                token += ch

            elif ch == 'l': # capture label

                state = 25
                token += ch

            elif ch == 'j': # capture jmp command

                state = 26
                token += ch

            elif ch == 'c': # catch cmp-command

                state = 29
                token += ch

            elif ch == ';': # capture comment 

                state = 33

            
            elif ch == '"': # catch a string

                state = 34
                # without "

            elif ch.isupper(): # capture identifier

                state = 35
                token += ch

            elif ch == 'd': # capture db keyword

                state = 36
                token += ch

            elif ch == "$": # catch variable with prefix $

                state = 38
                # not catching $

            elif ch == '_': # catch label for subprogram

                state = 40
                # not catch the character _

            elif ch == 'r': # catch ret-command

                state = 44
                token += ch
                
            else:   # other characters like space-characters etc.

                state = 0
                token = ""
        
        elif state == 1:    # state 1

            if ch == 'o':

                state = 2
                token += ch

            elif ch == 'u':

                state = 47
                token += ch

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 2: # state 2

            if ch == 'v':

                state = 3
                token += 'v'

            else: # error case
                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 3: # state 3

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 4:    # state 4
            if (ch >= 'a' and ch <= 'd'):

                state = 5
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 5:    # state 5
            if ch == 'x':
                state = 13
                token += ch
            
            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 6:    # state 6

            if ch.isdigit():

                state = 6
                token += ch

            elif ch.isspace():

                state = 0
                tokens.append(Token(token,"value"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 7:    # state 7

            if ch == 'd':

                state = 8
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 8:    # state 8

            if ch == 'd':

                state = 9
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 9:     # state 9

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 10:   # state 10

            if ch == 'u':

                state = 11
                token += ch

            else:   # error case
                
                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 11:   # state 11

            if ch == 'b':

                state = 12
                token += ch

            else:    # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 12:   # state 12

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 13:   # state 13

            if ch == ',' or ch.isspace():

                state = 0
                tokens.append(Token(token,"register"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 14:    # state 14

            if ch == 'n':

                state = 15
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 15:   # state 15

            if ch == 't':

                state = 16
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 16:   # state 16

            if ch.isspace():

                state = 0
                tokens.append(Token(token, "command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 17: # state 17

            if ch == 'x':

                state = 18
                token += ch

            elif ch.isspace():

                state = 0
                tokens.append(Token(token, "value"))
                token = ""

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 18:   # state 18

            if ch.isdigit() or (ch >= 'a' and ch <= 'f'):

                state = 18
                token += ch

            elif ch.isspace():

                state = 0
                tokens.append(Token(token,"value"))
                token = ""

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 19:   # state 19

            if ch == 'u':

                state = 20
                token += ch

            elif ch == 'o':

                state = 23
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 20:   # state 20

            if ch == 's':

                state = 21
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 21:   # state 21

            if ch == 'h':

                state = 22
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 22:   # state 22

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 23:   # state 23

            if ch == 'p':

                state = 24
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 24:   # state 24

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 25:   # state 25

            if ch.isdigit():

                state = 25
                token += ch

            elif ch == ':' or ch.isspace():

                state = 0
                tokens.append(Token(token,"label"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 26:   # state 26

            if ch == 'm':

                state = 27
                token += ch

            elif ch == 'e':     # catch je command 

                state = 32
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 27:   # state 27

            if ch == 'p':

                state = 28
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 28:   # state 28

            if ch.isspace():

                state = 0
                tokens.append(Token(token, "command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 29:   # state 29

            if ch == 'm':

                state = 30
                token += ch

            elif ch == 'a': # catch call-command

                state = 41
                token += ch

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 30:   # state 30

            if ch == 'p':

                state = 31
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 31:   # state 31

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 32:   # state 32

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 33:   # state 33

            if ch.isdigit() or ch.isalpha() or (ch.isspace() and ch != '\n') \
            or ch == '"':

                state = 33
                

            elif ch == '\n':

                state = 0

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 34:   # state 34

            if ch.isdigit() or ch.isalpha() or ch.isspace():

                state = 34
                token += ch

            elif ch == '"':

                state = 0
                tokens.append(Token(token,"string"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 35:   # state 35

            if ch.isdigit() or ch.isupper():

                state = 35
                token += ch

            elif ch == ' ' or ch == '\n':

                state = 0
                tokens.append(Token(token,"identifier"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 36:   # state 36

            if ch == 'b':

                state = 37
                token += ch

            elif ch == 'i':

                state = 49
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 37:   # state 37

            if ch.isspace():

                state = 0
                tokens.append(Token (token, "command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 38:   # state 38

            if ch.isalpha():

                state = 39
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 39:   # state 39

            if ch.isalpha() or ch.isdigit():

                state = 39
                token += ch

            elif ch.isspace():

                state = 0
                tokens.append(Token (token,"identifier"))
                token = ""

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 40:   # state 40

            if (ch >= 'a' and ch <='z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9'):

                state = 40
                token += ch

            elif ch == ':' or ch.isspace():
                
                state = 0
                tokens.append(Token (token,"subprogram"))
                token = ""

            else: # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 41:   # state 41

            if ch == 'l':

                state = 42
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 42:   # state 42

            if ch == 'l':

                state = 43
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 43:   # state 43

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 44:   # state 44

            if ch == 'e':

                state = 45
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 45:   # state 45

            if ch == 't':

                state = 46
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 46:   # state 46

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 47:   # state 47

            if ch == 'l':

                state = 48
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 48:   # state 48

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""
                

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()

        elif state == 49:   # state 49

            if ch == 'v':

                state = 50
                token += ch

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()


        elif state == 50:   # state 50

            if ch.isspace():

                state = 0
                tokens.append(Token(token,"command"))
                token = ""
                

            else:   # error case

                state = 0
                token = ""
                raise InvalidSyntax()





def scan():
    """
        scan: applys function scanner() to each line of the source code.
    """
    global lines
    assert len(lines) > 0 , "no lines"
    for line in lines:
        try:
            scanner(line)
        except InvalidSyntax:
            print "line=",line
    
def parser():
    """
        parser: parses the tokens of the list 'tokens'
    """

    global tokens
    global eax,ebx,ecx,edx

    assert len(tokens) > 0 , "no tokens"

    pointer = 0 # pointer for tokens
    token = Token("","")
    tmpToken = Token("","")

    while pointer < len(tokens):

        token = tokens[pointer]

        if token.token == "mov":      # mov commando

            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found argument!"
                return

            # TODO use token.t for this stuff
            if token.t == "register":

                tmpToken = token

                # it must follow a value / string / register / variable
                if pointer+1 < len(tokens):
                    pointer += 1
                    token = tokens[pointer]
                else:
                    print "Error: Not found argument!"
                    return

                # converts the token into float, if token contains olny digits.
                # TODO response of float
                if token.t == "identifier":     # for variables

                    # check of exists of variable
                    if token.token in variables:
                        token.token = variables[token.token]
                    else:
                        print "Error: undefine variable! --> " + token.token
                        return
                elif token.t == "string":
                    pass
                elif isinstance(token.token, float):
                    pass
                elif token.token.isdigit():
                    token.token = float(token.token)
                elif token.token[0] == '-' and token.token[1:].isdigit():
                    token.token = float(token.token[1:])
                    token.token *= -1
                elif token.t == "register": # loads out of register
                    if token.token == "eax":
                        token.token = eax
                    elif token.token == "ebx":
                        token.token = ebx
                    elif token.token == "ecx":
                        token.token = ecx
                    elif token.token == "edx":
                        token.token = edx



                if tmpToken.token == "eax":
                    eax = token.token
                elif tmpToken.token == "ebx":
                    ebx = token.token
                elif tmpToken.token == "ecx":
                    ecx = token.token
                elif tmpToken.token == "edx":
                    edx = token.token

            else:

                print "Error: No found register!"
                return

        elif token.token == "add":    # add commando

            pointer += 1
            token = tokens[pointer]

            if token.t == "register":

                tmpToken = token

                if pointer+1 < len(tokens):
                    pointer += 1
                    token = tokens[pointer]
                else:
                    print "Error: Not found number!"
                    return


                # converts the token into float, if token contains olny digits.
                if token.t == "register":
                    
                    # for the case that token is register
                    if token.token == "eax":
                        token.token = eax
                    elif token.token == "ebx":
                        token.token = ebx
                    elif token.token == "ecx":
                        token.token = ecx
                    elif token.token == "edx":
                        token.token = edx

                elif token.token.isdigit():
                    token.token = float(token.token)
                elif token.token[0] == '-' and token.token[1:].isdigit():
                    token.token = float(token.token[1:])
                    token.token *= -1
                else:
                    print "Error: ", token," is not a number!"
                    return



                if tmpToken.token == "eax":
                    eax += token.token

                    # update zero flag
                    if eax == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "ebx":
                    ebx += token.token

                    # update zero flag
                    if ebx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "ecx":
                    ecx += token.token

                    # update zero flag
                    if ecx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "edx":
                    edx += token.token

                    # update zero flag
                    if edx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                    


            else:

                print "Error: No found register!"
                return

        elif token.token == "sub":  # sub commando

            pointer += 1
            token = tokens[pointer]

            if token.t == "register":

                tmpToken = token

                if pointer+1 < len(tokens):
                    pointer += 1
                    token = tokens[pointer]
                else:
                    print "Error: Not found number!"
                    return

                # converts the token into float, if token contains olny digits.
                if token.t == "register":

                    # for the case that token is register
                    if token.token == "eax":
                        token.token = eax
                    elif token.token == "ebx":
                        token.token = ebx
                    elif token.token == "ecx":
                        token.token = ecx
                    elif token.token == "edx":
                        token.token = edx
                
                elif isinstance(token.token, float):
                    pass
                elif token.token.isdigit():
                    token.token = float(token.token)
                elif token.token[0] == '-' and token.token[1:].isdigit():
                    token.token = float(token.token[1:])
                    token.token *= -1
                else:
                    print "Error: ", token.token," is not a number!"
                    return


                if tmpToken.token == "eax":
                    eax -= token.token

                    # update zero flag
                    if eax == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "ebx":
                    ebx -= token.token

                    # update zero flag
                    if ebx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "ecx":
                    ecx -= token.token

                    # update zero flag
                    if ecx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False
                elif tmpToken.token == "edx":
                    edx -= token.token

                    # update zero flag
                    if edx == 0:
                        zeroFlag = True
                    else:
                        zeroFlag = False

            else:

                print "Error: No found register!"
                return

        elif token.token == "int": # int commando 

            tmpToken = token

            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found argument!"
                return

            if token.token == "0x80": # system interrupt 0x80
                
                if eax == 1:    # exit program

                    if ebx == 0:
                        print "END PROGRAM"
                        return
                    else:
                        print "END PROGRAM WITH ERRORS"
                        return

                elif eax == 3:

                    ecx = float(input(">> "))

                elif eax == 4: # output informations

                    print ecx

                    
        elif token.token == "push":   # push commando

            tmpToken = token

            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found register!"
                return

            # pushing register on the stack
            if token.token == "eax":

                stack.append(eax)

            elif token.token == "ebx":

                stack.append(ebx)

            elif token.token == "ecx":

                stack.append(ecx)

            elif token.token == "edx":

                stack.append(edx)


        elif token.token == "pop":   # pop commando

            tmpToken = token

            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found register!"
                return

            # pop register from stack
            if token.token == "eax":

                eax = stack.pop()

            elif token.token == "ebx":

                ebx = stack.pop()

            elif token.token == "ecx":

                ecx = stack.pop()

            elif token.token == "edx":

                edx = stack.pop()

        elif token.t == "label":        # capture label

            jumps[token.token] = pointer

        elif token.token == "jmp":      # capture jmp command

            # it must follow a label
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found label!"
                return

            if token.t == "label":

                pointer = jumps[token.token] 

            else:
                print "Error: expected a label!"


        elif token.token == "cmp":
            # TODO
            
            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer]
            else:
                print "Error: Not found argument!"
                return

            if token.t == "register":

        
                # it must follow a register
                if pointer+1 < len(tokens):
                    pointer += 1
                    tmpToken = tokens[pointer] # next register
                else:
                    print "Error: Not found register!"
                    return

                # actual comparing
                if token.token == "eax":

                    if tmpToken.token == "eax":

                        if eax == eax:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ebx":

                        if eax == ebx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ecx":

                        if eax == ecx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "edx":

                        if eax == edx:
                            zeroFlag = True
                        else:
                            zeroFlag = False


                elif token.token == "ebx":

                    if tmpToken.token == "eax":

                        if ebx == eax:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ebx":

                        if ebx == ebx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ecx":

                        if ebx == ecx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "edx":

                        if ebx == edx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                
                elif token.token == "ecx":

                    if tmpToken.token == "eax":

                        if ecx == eax:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ebx":

                        if ecx == ebx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ecx":

                        if ecx == ecx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "edx":

                        if ecx == edx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                
                elif token.token == "edx":

                    if tmpToken.token == "eax":

                        if edx == eax:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ebx":

                        if edx == ebx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "ecx":

                        if edx == ecx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

                    elif tmpToken.token == "edx":

                        if edx == edx:
                            zeroFlag = True
                        else:
                            zeroFlag = False

            
            else:
                print "Error: Not found register!"
                return

        
        elif token.token == "je":

            # it must follow a label
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer] # next register
            else:
                print "Error: Not found argument"
                return

            # check of label
            if token.t == "label":

                # actual jump
                if zeroFlag:

                    pointer = jumps[token.token]

            else:

                 print "Error: Not found label"
                 return


        elif token.t == "identifier":

            # check whether identifier is in variables-table
            if token.token not in variables:

                # it must follow a command
                if pointer+1 < len(tokens):
                    pointer += 1
                    tmpToken = tokens[pointer] # next register
                else:
                    print "Error: Not found argument"
                    return

                if tmpToken.t == "command" and tmpToken.token == "db":

                    # it must follow a value (string)
                    if pointer+1 < len(tokens):
                        pointer += 1
                        tmpToken = tokens[pointer] # next register
                    else:
                        print "Error: Not found argument"
                        return

                    if tmpToken.t == "value" or tmpToken.t == "string":

                        if tmpToken.t == "value":
                            variables[token.token] = float(tmpToken.token)
                        elif tmpToken.t == "string":
                            variables[token.token] = tmpToken.token
                

                else:

                    print "Error: Not found db-keyword"
                    return

        elif token.token == "call":     # catch the call-command

            # it must follow a subprogram label
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer] # next register
            else:
                print "Error: Not found subprogram label"
                return


            if token.t == "subprogram":

                if token.token in jumps:

                    # save the current pointer
                    returnStack.append(pointer)   # eventuell pointer + 1
                    # jump to the subprogram
                    pointer = jumps[token.token]
                
                else: # error case

                    print "Error: Unknow subprogram!"
                    return

            else:   # error case

                print "Error: Not found subprogram"
                return

        
        elif token.token == "ret":  # catch the ret-command

            if len(returnStack) >= 1:

                pointer = returnStack.pop() 

            else: # error case

                print "Error: No return adress on stack"
                return

        elif token.t == "subprogram":

            pass

        
        elif token.token == "mul":      # catch mul-command

            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer] # next register
            else:
                print "Error: Not found argument"
                return

            if token.t == "register":

                if token.token == "eax":

                    eax *= eax

                elif token.token == "ebx":

                    eax *= ebx

                elif token.token == "ecx":

                    eax *= ecx

                elif token.token == "edx":

                    eax *= edx

            else:

                print "Error: Not found register"
                return



        elif token.token == "div":

            # it must follow a register
            if pointer+1 < len(tokens):
                pointer += 1
                token = tokens[pointer] # next register
            else:
                print "Error: Not found argument"
                return

            if token.t == "register":

                if token.token == "eax":

                    eax /= eax

                elif token.token == "ebx":

                    eax /= ebx

                elif token.token == "ecx":

                    eax /= ecx

                elif token.token == "edx":

                    eax /= edx

            else:

                print "Error: Not found register"
                return



        # increment pointer for fetching next token.
        pointer += 1


def registerLabels():
    """
        This function search for labels / subprogram-labels and register this in the 'jumps' list.
    """
    for i in range(len(tokens)):
        if (tokens[i].t == "label"):
            jumps[tokens[i].token] = i
        elif tokens[i].t == "subprogram":
            jumps[tokens[i].token] = i
    

def resetInterpreter():
    """
        resets the interpreter mind.
    """
    global eax, ebx, ecx, edx, zeroFlag, stack
    global variables, jumps, lines, tokens, returnStack
    eax = 0
    ebx = 0
    ecx = 0
    edx = 0
    zeroFlag = False
    stack = []
    jumps = {}
    variables = {}
    lines = []
    tokens = []
    returnStack = []

# DEBUG FUNCTION
# def printTokens():
#     for token in tokens:
#         print token.token, " --> ", token.t


# main program
def main():
    """
        reads textfiles from the command-line and interprets them. 
    """

    # [1:] because the first argument is the program itself.
    for arg in sys.argv[1:]:

        resetInterpreter()  # resets interpreter mind

        try:

            loadFile(arg)
            scan()
            registerLabels()
            parser()

        except:

            print "Error: File %s not found!" % (arg)

        

    
    
if __name__ == "__main__":
    main()
