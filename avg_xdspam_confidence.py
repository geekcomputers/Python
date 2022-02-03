fh = open("mbox-short.txt")
# The 'mbox-short.txt' file can be downloaded from the link: https://www.py4e.com/code3/mbox-short.txt
sum = 0
count = 0
for fx in fh:
    fx = fx.rstrip()
    if not fx.startswith("X-DSPAM-Confidence:"):
        continue
    fy = fx[19:]
    count = count + 1
    sum = sum + float(fy)
print("Average spam confidence: ", sum / count)
