import time
import os
#Importing our Bindoer
print("To Kaise Hai Ap Log!")
time.sleep(1)
print("Chaliye Binod Karte Hai!")
def checkBinod(file):#Trying to find Binod In File Insted Of Manohar Ka Kotha
    with open(file, "r") as f:
        fileContent = f.read()
    if 'binod' in fileContent.lower():
        print(
            f'**************Congratulations Binod found in {f}********************')
        return True


if __name__ == '__main__':
    ans = False
    print("************binod Detector********************")
    dir_contents = os.listdir()
    for item in dir_contents:
        if item.endswith('txt'):
            ans = checkBinod(item)
            if(ans is False):
                print('Binod not found Try Looking In Manohar Ka Kotha!!')
