import cv2 as cv
img = cv.imread('./scenery.jpg')

# colorspace=> RGB, HSV, Gray and so on

# BGR to Grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray-image',gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV-image',hsv)

hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV->BGR',hsv_bgr)


lab_bgr = cv.cvtColor(hsv, cv.COLOR_LAB2BGR)
cv.imshow('LAB->BGR',lab_bgr)

cv.waitKey(0)
