import cv2 as cv
import numpy as np

img = cv.imread('./scenery.jpg')
cv.imshow('Original',img)

# create blank image
blank = np.zeros(img.shape[:2], dtype='uint8')
b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('blue-on-blank',blue)
cv.imshow('green-on-blank',green)
cv.imshow('red-on-blank',red)

#split the image into 3 color channels
b,g,r = cv.split(img)
cv.imshow('Blue',b)
cv.imshow('Green',g)
cv.imshow('Red',r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged-image',merged)

cv.waitKey(0)
