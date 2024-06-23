coverage = {
    "bal-case-1" : False,
    "bal-case-2" : False,
    "bal-case-3" : False,
    "bal-case-4" : False
}

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return self.items == []

    def peek(self):
        return self.items[-1]

    def display(self):
        return self.items


def is_same(p1, p2):
    if p1 == "(" and p2 == ")":
        return True
    elif p1 == "[" and p2 == "]":
        return True
    elif p1 == "{" and p2 == "}":
        return True
    else:
        return False


def is_balanced(check_string):
    s = Stack()
    index = 0
    is_bal = True
    while index < len(check_string) and is_bal:
        paren = check_string[index]
        if paren in "{[(":
            coverage["bal-case-1"] = True
            s.push(paren)
        else:
            if s.is_empty():
                coverage["bal-case-2"] = True
                is_bal = False
            else:
                coverage["bal-case-3"] = True
                top = s.pop()
                if not is_same(top, paren):
                    coverage["bal-case-4"] = True
                    is_bal = False
        index += 1

    if s.is_empty() and is_bal:
        coverage["bal-case-5"] = True
        return True
        
    else:
        coverage["bal-case-6"] = True
        return False
    
def bal_case_two_test():
    empty_str = " "
    assert(is_balanced(empty_str) == False)
    print("case-two-test passed OK")


def resetDic():
    for i in coverage.keys():
        coverage[i] = False

def printCov():
    for branch, hit in coverage.items():
        if hit:
            print(branch, "was hit")
        else:
            print(branch, "was not hit")
    resetDic()

print(is_balanced("[((())})]"))
bal_case_two_test()
printCov()


