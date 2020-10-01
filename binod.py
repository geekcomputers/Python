import os    #The OS module in python provides functions for interacting with the operating system


def checkBinod(file):       #this function will check there is any 'Binod' text in file or not
    with open(file, "r") as f: #we are opening file in read mode and using 'with' so need to take care of close()
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
                print('Binod not found!')
