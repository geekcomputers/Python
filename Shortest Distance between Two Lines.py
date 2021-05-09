import math
import numpy as NP

LC1=eval(input("Enter DRs of Line 1 : "))
LP1=eval(input("Enter Coordinate through which Line 1 passes : "))
LC2=eval(input("Enter DRs of Line 2 : "))
LP2=eval(input("Enter Coordinate through which Line 2 passes : "))
a1,b1,c1,a2,b2,c2=LC1[0],LC1[1],LC1[2],LC2[0],LC2[1],LC2[2]
x=NP.array([[LP2[0]-LP1[0],LP2[1]-LP1[1],LP2[2]-LP1[2]],[a1,b1,c1],[a2,b2,c2]])
y=math.sqrt((((b1*c2)-(b2*c1))**2)+(((c1*a2)-(c2*a1))**2)+(((a1*b2)-(b1*a2))**2))
