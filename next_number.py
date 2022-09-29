x, li, small, maxx, c = input(), [], 0, 0, 1
li.extend(int(x[i]) for i in range(len(x)))
for i in range(len(li) - 1, -1, -1):
    if i == 0:
        print("No Number Possible")
        c = 0
        break
    if li[i] > li[i - 1]:
        small = i - 1
        maxx = i
        break
for i in range(small + 1, len(li)):
    if li[i] > li[small] and li[i] < li[maxx]:
        maxx = i
li[small], li[maxx] = li[maxx], li[small]
li = li[: small + 1] + sorted(li[small + 1 :])
if c:
    for item in li:
        print(item, end="")
