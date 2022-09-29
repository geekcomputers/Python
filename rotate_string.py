def left_rotate(s, val):
    s1 = s[:val]
    s2 = s[val:]
    return s2 + s1


def right_rotate(s, val):
    s1 = s[:len(s) - val]
    s2 = s[len(s) - val :]
    return s2 + s1


def circular_rotate(s):
    s = list(s)
    mid = len(s) // 2
    for idx, i in enumerate(reversed(range(mid, len(s)))):
        s[idx], s[i] = s[i], s[idx]
    return s


s = "aditya"
print("".join(circular_rotate(s)))
