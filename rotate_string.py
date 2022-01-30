def left_rotate(s, val):
    s1 = s[0:val]
    s2 = s[val:]
    return s2 + s1


def right_rotate(s, val):
    s1 = s[0 : len(s) - val]
    s2 = s[len(s) - val :]
    return s2 + s1


def circular_rotate(s):
    s = list(s)
    idx = 0
    mid = len(s) // 2
    for i in reversed(range(mid, len(s))):
        s[idx], s[i] = s[i], s[idx]
        idx += 1
    return s


s = "aditya"
print("".join(circular_rotate(s)))
