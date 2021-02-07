#--- Program Description ---
#classifies stop sign by looking for color red and octagon shape

import numpy as np
import cv2 as cv

print("OpenCv Version:",cv.__version__)

#consts used for calculating distance to stop sign
STOP_SIGN_LENGTH = 0.71  #m (it is 75cm across with 2cm white borders on each side)
VIEW_ANGLE_HORIZ = 52 #deg

#define detect shape function (input contour obj)
def detect_shape(c):
    # Compute perimeter of contour and perform contour approximation
    shape = ""
    perim = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.005 * perim, True)

    #if the contour is small set shape = "too_small"
    minRect = cv.minAreaRect(c) 
    if minRect[1][0] < 40.0 or minRect[1][1] < 40.0:
        shape = "too_small"

    # Triangle
    elif len(approx) == 3:
        shape = "triangle"

    # Square or rectangle
    elif len(approx) == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)

        # A square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # Pentagon
    elif len(approx) == 5:
        shape = "pentagon"

    # Hexagon
    elif len(approx) == 6:
        shape = "hexagon"

    # Octagon 
    elif len(approx) == 8:
        shape = "octagon"

    # Star
    elif len(approx) == 10:
        shape = "star"

    # Otherwise assume as circle or oval
    else:
        shape = "circle"

    return shape
    

#read in image and get W,H
stopSignImg = cv.imread('stopSign1.png')
W = len(stopSignImg[1])
H = len(stopSignImg)
print("Width: ",W, "\nHeight: ",H)


#convert image to HSV (Hue, Saturation, Value) for easier color detection
stopSignHsv = cv.cvtColor(stopSignImg, cv.COLOR_BGR2HSV)
#print(stopSignHsv[655][560])

# generate 2 masks for red color (1 lower and 1 upper) and combine them into 1 mask
maskRedLow = cv.inRange(stopSignHsv, (0,100,70), (8,255,255))
maskRedHigh = cv.inRange(stopSignHsv, (165,100,70), (180,255,255))
mask = cv.bitwise_or(maskRedLow, maskRedHigh)

# apply the mask to the original image (gets only the pixels which meet the mask req)
detectRed = cv.bitwise_and(stopSignImg, stopSignImg, mask=mask)
detectRed[(mask==255)] = [0,0,255]

#convert to greyscale and then binary so we can draw contours on img
grayRedDetect = cv.cvtColor(detectRed, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(grayRedDetect, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

#Next we want to group the red sections and then look for edges of octogon
#if we find both red color and shape of octogon then we have a stop sign

#Find contours and detect shape
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#Draw all contour lines
cv.drawContours(detectRed, contours, -1, (0,255,0), 3)
cv.imshow("image", detectRed)
cv.waitKey()

#if any contour is an octagon then it is a stop sign
for idx,conts in enumerate(contours):
    shape = detect_shape(conts)

    if shape == "octagon":
        print("STOP SIGN DETECTED at ID:", idx)
        cv.drawContours(stopSignImg, contours, idx, (0,255,0), 2)

        #draw bounding rectangle around stop sign
        rect_x,rect_y,rect_w,rect_h = cv.boundingRect(conts)
        cv.rectangle(stopSignImg, (rect_x,rect_y), (rect_x+rect_w,rect_y+rect_h),(0,0,0),2)

        relAngleStopSign = ((rect_w/W)*VIEW_ANGLE_HORIZ)*(3.14159/180.0) #in rad
        distEstToSign = STOP_SIGN_LENGTH / relAngleStopSign
        print("Distance to Stop Sign:",distEstToSign)

#display image
cv.imshow("image", stopSignImg)
cv.waitKey()