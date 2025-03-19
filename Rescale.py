import cv2 as cv

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture('cat-video(2).mp4')

while True:
    isTrue, frame = capture.read()
    
    if not isTrue:  # Stop if video ends
        break  

    frame_resized = rescaleFrame(frame)

    cv.imshow("Original Video", frame)  # Corrected
    cv.imshow("Resized Video", frame_resized)  # Corrected
    
    if cv.waitKey(20) & 0xFF == ord('d'):  # Press 'd' to exit
        break

capture.release()
cv.destroyAllWindows()
