import cv2 as cv

#Read the image
img = cv.imread('scenery.jpg')

#converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Blur the image
blur1 = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur1', blur1)

blur2 = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur2', blur2)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)

# Dilating the image
dialated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dialated)

#Eroding
eroded = cv.erode(dialated, (7,7), iterations=1)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[20:50, 10:40]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()