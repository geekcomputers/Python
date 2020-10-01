import os

# function to check if 'binod' is present in the file.
def checkBinod(file):
    with open(file, "r") as f:
        fileContent = f.read()
    if 'binod' in fileContent.lower():
        print(
            f'**************Congratulations Binod found in {f}********************')
        return True
    else:
        return False


if __name__ == '__main__':
    print("************binod Detector********************")
    dir_contents = os.listdir()
    for item in dir_contents:
        if item.endswith('txt'):
            ans = checkBinod(item)
            if(ans is False):
                print('Binod not found!')
