import cv2 as cv
import numpy as np

img = cv.imread('scenery.jpg')

cv.imshow('Scenery', img)

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x => Left
# -y => Up
# x => Right
# y => Down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (hetght, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, hetght//2)
    rotmat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, hetght)
    return cv.warpAffine(img, rotmat, dimensions)

# angle => Positive => Counter Clockwise
# angle => Negative => Clockwise

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)


# Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping
# 0 => Vertical Flip
# -1 => Horizontal and Vertical Flip
cv.flip(img, -1)
cv.imshow('Flip', img)

cv.waitKey(0)