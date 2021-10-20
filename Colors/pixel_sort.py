'''Pixel Sorting'''

#Importing Libraries
import cv2
import numpy as np
import math 
import colorsys
import pandas as pd
import os 
import argparse 
from tqdm import tqdm

#Importing the external file Library 
import sound 

#Taking arguments from command line
parser = argparse.ArgumentParser() #you iniatize as such
parser.add_argument("-f", required=True, help="enter fileName of your picture")
#parser.add_argument("-s", required=True, help="Speed factor of the audio to be increased or decreased")
#parser.add_argument("-av", required=True, help="Speed factor of the audio visualizer to be increased or decreased")

#the add_argument tells you what needs to be given as an input sp its help 
args = parser.parse_args() #you take the arguments from command line

os.makedirs("Image_sort/"+str(args.f))
print(str(args.f).capitalize()+" directory is created.") 

#Defining all global variables 
df = []
total = 0
dict , final , img_list = {} , [] , []

#Create dataframe and save it as an excel file
def createDataSet(val = 0 , data = []) : 
         global dict 
         dict[len(data)] = data 
         if val != 0 : 
             if val == max(dict.keys()) : 
                 final_df = pd.DataFrame(dict[val],columns=["Blue","Green","Red"]) 
                 final_df.to_excel("Image_sort/"+str(args.f)+"/"+"output.xlsx") 
    

#Generating colors for each row of the frame 
def generateColors(c_sorted,frame,row) :
    global df , img_list
    height = 25   
    img = np.zeros((height,len(c_sorted),3),np.uint8)  
    for x in range(0,len(c_sorted)) :
        r ,g , b = c_sorted[x][0]*255,c_sorted[x][1]*255,c_sorted[x][2]*255
        c =  [r,g,b]
        df.append(c)
        img[:,x] = c #the color value for the xth column , this gives the color band 
        frame[row,x] = c #changes added for every row in the frame 
              
    createDataSet(data = df) 
    return img , frame

#Measures the total number of pixels that were involved in pixel sort
def measure(count,row,col,height,width) : 
    global total 
    total += count 
    if row==height-1 and col == width-1 : 
        createDataSet(val = total)

#Step Sorting Algorithm
def step (bgr,repetitions=1):
    b,g,r = bgr
    #lum is calculated as per the way the humans view the colors
    lum = math.sqrt( .241 * r + .691 * g + .068 * b )
    
    #conversion of rgb to hsv values 
    h, s, v = colorsys.rgb_to_hsv(r,g,b) # h,s,v is a better option for classifying each color 
    
    #Repetitions are taken to decrease the noise 
    h2 = int(h * repetitions)
    v2 = int(v * repetitions)
    
    #To get a smoother color band 
    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum
 
    return h2, lum, v2

#Threshold set for avoiding extreme sorting of the pixels 
def findThreshold(lst , add) : 
    for i in lst  : 
        add.append(sum(i))
    return (max(add)+min(add))/2

def makeVideo() : 
    out = cv2.VideoWriter("Image_sort/"+str(args.f)+"/"+str(args.f)+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), 16, (800,500))
    for count in tqdm(range(1,500+1)) :
        fileName = "Image_sort/"+str(args.f)+"/"+str(count)+".jpg"
        img = cv2.imread(fileName) 
        out.write(img) 
        os.remove(fileName) 
    out.release()

def main() : 
    global img_list 
    img = cv2.imread("Image/"+str(args.f)+".jpg")
    img = cv2.resize(img,(800,500))  
    img_list.append(img) 
    
    height , width , _ = img.shape
    print(">>> Row-wise Color sorting")
    for row in tqdm(range(0,height)) : 
        color , color_n = [] , [] 
        add = []
        
        for col in range(0,width) : 
            val = img[row][col].tolist()
            
            #val includes all rgb values between the range of 0 to 1 
            #This makes the sorting easier and efficient 
            val = [i/255.0 for i in val] 
            color.append(val)               
        
        thresh = findThreshold(color, add) #setting the threshold value for every row in the frame
        
        #For the specific row , if all the values are non-zero then it is sorted with color
        if np.all(np.asarray(color)) ==  True : 
                    color.sort(key=lambda bgr : step(bgr,8)) #step sorting
                    band , img = generateColors(color,img,row)
                    measure(len(color),row,col,height,width)
                    
        #For the specific row , if any of the values are zero it gets sorted with color_n
        if np.all(np.asarray(color)) == False :    
                    for ind , i in enumerate(color) : 
                        #Accessing every list within color 
                        #Added to color_n if any of the element in the list is non-zero 
                        #and their sum is less than threshold  value
                        
                        if np.any(np.asarray(i)) == True and sum(i) < thresh : 
                            color_n.append(i) 
                            
                    color_n.sort(key=lambda bgr : step(bgr,8)) #step sorting
                    band , img = generateColors(color_n,img,row)
                    measure(len(color_n),row,col,height,width)
        cv2.imwrite("Image_sort/"+str(args.f)+"/"+str(row+1)+".jpg" , img)
       
    #Writing down the final sorted image
    cv2.imwrite("Image_sort/"+str(args.f)+"/"+str(args.f)+".jpg",img) #Displaying the final picture
    
    print("\n>>> Formation of the Video progress of the pixel-sorted image")
    makeVideo()
    sound.main(args.f) #Calling the external python file to create the audio of the pixel-sorted image
    
main()