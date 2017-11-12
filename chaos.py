# A simple program illustrating chaotic behaviour

def main():
    print ("This program illustrates a chaotic function")
    x = float((input("Enter a number between 0 and 1: ")))
    for i in range(10):
        x = 3.9 * x * (1-x)
        print (x)

if __name__ == '__main__':
    main()
