# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import time

import cv2
import numpy as np

## Preparation for writing the ouput video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

##reading from the webcam
cap = cv2.VideoCapture(0)

## Allow the system to sleep for 3 seconds before the webcam starts
time.sleep(3)
count = 0
background = 0

## Capture the background in range of 60
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)


## Read every frame from the webcam, until the camera is open
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)

    ## Convert the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Generat masks to detect red color
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2
