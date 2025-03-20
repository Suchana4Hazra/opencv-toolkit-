import cv2 as cv
import numpy as np

img = cv.imread('./poet.jpg')

cv.imshow('cat', img)

#create a blank image
blank = np.zeros(img.shape[:2],dtype='uint8')
cv.imshow('blank-image',blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray-cat', gray)

# to reduce no of contours
# blur = cv.GaussianBlur(gray,(11,11),cv.BORDER_DEFAULT)
# cv.imshow('blur',blur)

# canny = cv.Canny(img, 125, 175)
# cv.imshow('canny',canny)

# threshold
ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY) #125px-255px white else black
cv.imshow('Thresh', thresh)
# find contour
contours,  hierarchies = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found!')

# draw contours on the blank image
cv.drawContours(blank, contours, -1, (255,25,90),thickness=2)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)