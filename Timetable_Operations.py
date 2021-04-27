##Clock in pt2thon##

t1 = input("Init schedule : ") # first schedule
HH1 = int(t1[0]+t1[1])
MM1 = int(t1[3]+t1[4])
SS1 = int(t1[6]+t1[7])

t2 = input("Final schedule : ") # second schedule
HH2 = int(t2[0]+t2[1])
MM2 = int(t2[3]+t2[4])
SS2 = int(t2[6]+t2[7])

tt1 = (HH1*3600)+(MM1*60)+SS1 # total schedule 1
tt2 = (HH2*3600)+(MM2*60)+SS2 # total schedule 2
tt3 = tt2-tt1                 # difference between tt2 e tt1

# Part Math
if (tt3 < 0):
    # If the difference between tt2 e tt1 for negative :

	a = 86400 - tt1 # 86400 is seconds in 1 day;
	a2 = a + tt2   # a2 is the difference between 1 day e the <hours var>;
	Ht = a2//3600 # Ht is hours calculated;

	a = a2 % 3600 # Convert 'a' in seconds;
	Mt = a//60   # Mt is minutes calculated;
	St = a % 60 # St is seconds calculated;

else:
    # If the difference between tt2 e tt1 for positive :

	Ht = tt3//3600     # Ht is hours calculated;
	z = tt3 % 3600    # 'z' is tt3 converting in hours by seconds

	Mt = z//60      # Mt is minutes calculated;
	St = tt3 % 60  # St is seconds calculated;

# special condition below : 
if (Ht < 10):
	h = '0'+ str(Ht)
	Ht = h
if (Mt < 10):
	m = '0'+ str(Mt)
	Mt = m
if (St < 10):
	s = '0'+ str(St)
	St = s
# add '0' to the empty spaces (caused by previous operations) in the final result!

print("final result is :", str(Ht)+":"+str(Mt)+":"+str(St)) # final result (formatted in clock)