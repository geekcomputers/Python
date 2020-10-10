import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

str = ''
faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('Resources/keras_model.h5')
data = np.ndarray(shape=(1, 224,224, 3), dtype=np.float32)


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    success,img = cap.read()
  
    cv2.imshow("webcam", img)
    faces = faceCascade.detectMultiScale(img,1.1,4)
    
    
    for (x,y,w,h) in faces:
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img,(224,224))
        normalized_image_array = (crop_img.astype(np.float32) / 127.0) - 1    
        data[0] = normalized_image_array    
        prediction = model.predict(data)
        print(prediction)
        if prediction[0][0] > prediction[0][1]:
            str = 'Mask'
        else:
            str = 'Without-mask'


        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,str,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
        cv2.imshow("Result", img)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
