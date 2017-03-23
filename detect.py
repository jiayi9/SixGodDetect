#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

t0 = time.time()


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/media/sf_share/new_3.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = gray.copy()[100:1300,300:1500]
#
#img = img.copy()[100:1300,300:1500]

#cv2.imshow('image',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imwrite("/media/sf_share/cropped.jpg",gray)
#
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#ret, thresh = cv2.threshold(gray,60,255,cv2.THRESH_OTSU)

ret, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)

cv2.imwrite("/media/sf_share/thresh.jpg",thresh)


#cv2.imshow('image',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# noise removal
kernel = np.ones((15,15),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

kernel = np.ones((3,3),np.uint8)

opening = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)

cv2.imwrite("/media/sf_share/opening.jpg",opening)


#cv2.imshow('image',opening)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# to find tube, dilate

dilate_height = 200

dilate_width = 50

kernel = np.ones((dilate_height,dilate_width),np.uint8)

dilated = cv2.dilate(opening, kernel)

#cv2.imshow('image', dilated)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


    # find tubes
    
_,contours,hierarchy = cv2.findContours(dilated, 1, 2)
drawing = img.copy()
opening_copy = opening.copy()

L = []
BALLS = []

for index, contour in enumerate(contours):
    print(index)
    
    x,y,w,h = cv2.boundingRect(contour)
    drawing = cv2.rectangle(drawing,(x,y),(x+w,y+h),(0,240,0),3)
    
    crop = opening_copy[y:y+h,x:x+w]
#    cv2.imshow('image',crop)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    crop_erode_height = 10
    crop_erode_width = 15
    
#    kernel = np.ones((crop_erode_height,crop_erode_width),np.uint8)
#    crop_eroded = cv2.erode(crop, kernel, iterations=1)
    
    dist_transform = cv2.distanceTransform(crop,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.55*dist_transform.max(),255,0)
    
    cv2.imwrite("/media/sf_share/tmp.jpg",sure_fg)
    tmp = cv2.imread("/media/sf_share/tmp.jpg",0)
    ret, tmp2 = cv2.threshold(tmp,80,255,cv2.THRESH_BINARY)
#    
#    cv2.imshow('image',tmp2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    _,small_contours,_ = cv2.findContours(tmp2, 1, 2)

    for ball in small_contours:
        bx,by,bw,bh = cv2.boundingRect(ball)
        BALLS.append([x+bx+int(round(bw/2)), int(round(y+by+bh/2))])
    
    
    L.append([x,y,w,h, len(small_contours)])    
    
    
#    cv2.imshow('image',crop_eroded)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
 

font = cv2.FONT_HERSHEY_SIMPLEX

for index, vector in enumerate(L):
    x = vector[0]
    y = vector[1]-20
    text = str(vector[4])
    print(x,y,text)
    cv2.putText(drawing,text,(x,y), font, 4,(0,0,255),2,cv2.LINE_AA)
    
for index, ball in enumerate(BALLS):
    x = ball[0]
    y = ball[1]
    print(x,y)
    cv2.ellipse(drawing,(x,y),(10,10),0,0,360,255,-1)

    
#cv2.imshow('image',drawing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


cv2.imwrite("/media/sf_share/count.jpg",drawing)



compare = np.concatenate((img, drawing), axis=1)


cv2.imwrite("/media/sf_share/compare.jpg",compare)


t1 = time.time()

total = t1 -t0

total
