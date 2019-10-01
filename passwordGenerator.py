#PasswordGenerator GGearing314 01/10/19
from random import *
case=randint(1,2)
number=randint(1,99)
animals=("ant","alligator","baboon","badger","barb","bat","beagle","bear","beaver","bird","bison","bombay","bongo","booby","butterfly","bee","camel","cat","caterpillar","catfish","cheetah","chicken","chipmunk","cow","crab","deer","dingo","dodo","dog","dolphin","donkey","duck","eagle","earwig","elephant","emu","falcon","ferret","fish","flamingo","fly","fox","frog","gecko","gibbon","giraffe","goat","goose","gorilla")
colour=("red","orange","yellow","green","blue","indigo","violet","purple","magenta","cyan","pink","brown","white","grey","black")
chosenanimal= animals[randint(0,len(animals))]
chosencolour=colour[randint(0,len(colour))]
if case==1:
    chosenanimal=chosenanimal.upper()
    print(chosencolour,number,chosenanimal)
else:
    chosencolour=chosencolour.upper()
    print(chosenanimal,number,chosencolour)
#print("This program has exatly ",(len(animals)*len(colour)*99*2),"different combinations") #I'm not sure this is right
input("Press enter to close...")

