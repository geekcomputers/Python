def gcd1(a,b):
  if not a:return b
  return gcd1(b%a,a)


def gcd2(a,b):
    while b:a,b=b,a%b
    return a

print(gcd1(30,15))
print(gcd2(30,15))
