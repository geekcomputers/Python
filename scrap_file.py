#Author : RIZWAN AHMAD 



#pip3 install requests 

import requests



#Function for download file parameter taking as url

def download(url):
    f = open('file_name.jpg','wb') # opening file in write binary('wb') mode with file_name.ext ext=extension 
    f.write(requests.get(url).content) # Writing File Content in file_name.jpg
    f.close()                          
    print("Succesfully Downloaded")

url='https://avatars0.githubusercontent.com/u/29729380?s=400&v=4' #URL from which we want to download
    
download(url) 
       