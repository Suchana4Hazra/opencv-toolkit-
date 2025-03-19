import cv2 as cv
import numpy as np

blank_img = np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank_img)

# 1. Paint the image a certain color
blank_img[:] = 0,255,0
cv.imshow('Green',blank_img)

# 2. Draw a rectangle
cv.rectangle(blank_img,(0,0),(250,250),(0,0,255),thickness=cv.FILLED)
cv.imshow('Rectangle',blank_img)

# 3. Draw a circle
cv.circle(blank_img,(250,250),40,(255,255,0),thickness=-1)   

# 4. Draw a line
cv.line(blank_img,(30,30),(250,250),(25,90,0),thickness=3)


# Put text on the image
cv.putText(blank_img,'Hello Suchana!',(100,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,255),2)
# Show the final image
cv.imshow('Final Image', blank_img)

cv.waitKey(0)
cv.destroyAllWindows()