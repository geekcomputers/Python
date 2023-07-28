# updating records in a binary file

import pickle


def update():

    with open("studrec.dat", "rb+") as File:
        value = pickle.load(File)
        found = False
        roll = int(input("Enter the roll number of the record"))
        
        for i in value:
            if roll == i[0]:
                print(f"current name {i[1]}")
                print(f"current marks {i[2]}")
                i[1] = input("Enter the new name")
                i[2] = int(input("Enter the new marks"))
                found = True

        if not found:
            print("Record not found")

        else:
            pickle.dump(value, File)
            File.seek(0)
            print(pickle.load(File))


update()
