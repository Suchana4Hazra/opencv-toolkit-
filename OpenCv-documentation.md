# OpenCV Documentation
## From Beginner to Advanced

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Python](#python)
- [Basics](#basics)
  - [Reading and Displaying Images](#reading-and-displaying-images)
  - [Writing Images](#writing-images)
  - [Reading and Writing Videos](#reading-and-writing-videos)
  - [Basic Image Operations](#basic-image-operations)
  - [Drawing Functions](#drawing-functions)
- [Image Processing](#image-processing)
  - [Color Spaces](#color-spaces)
  - [Image Arithmetic](#image-arithmetic)
  - [Thresholding](#thresholding)
  - [Smoothing and Blurring](#smoothing-and-blurring)
  - [Morphological Operations](#morphological-operations)
  - [Gradients and Edge Detection](#gradients-and-edge-detection)
  - [Image Pyramids](#image-pyramids)
  - [Contours](#contours)
  - [Histograms](#histograms)
- [Feature Detection and Description](#feature-detection-and-description)
  - [Harris Corner Detection](#harris-corner-detection)
  - [SIFT (Scale-Invariant Feature Transform)](#sift-scale-invariant-feature-transform)
  - [SURF (Speeded-Up Robust Features)](#surf-speeded-up-robust-features)
  - [ORB (Oriented FAST and Rotated BRIEF)](#orb-oriented-fast-and-rotated-brief)
  - [Feature Matching](#feature-matching)
- [Video Analysis](#video-analysis)
  - [Background Subtraction](#background-subtraction)
  - [Optical Flow](#optical-flow)
  - [Object Tracking](#object-tracking)

<!-- INTRODUCTION -->
## Introduction

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.

These algorithms can be used to:
- Detect and recognize faces
- Identify objects
- Classify human actions in videos
- Track camera movements
- Track moving objects
- Extract 3D models of objects
- Produce 3D point clouds from stereo cameras
- Stitch images together to produce a high resolution image of an entire scene
- Find similar images from an image database
- Remove red eyes from images taken using flash
- Follow eye movements
- Recognize scenery and establish markers to overlay it with augmented reality, etc.

The library is widely used in companies, research groups, and governmental bodies.

<!-- INSTALLATION -->
## Installation

### Windows

1. **Using pip (Python)**
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python  # For extra modules
   ```

### macOS

1. **Using Homebrew**
   ```bash
   brew install opencv
   ```

2. **Using pip (Python)**
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python  # For extra modules
   ```

### Linux

1. **Using package manager**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install libopencv-dev python3-opencv

   # Fedora
   sudo dnf install opencv opencv-devel
   ```

2. **Using pip (Python)**
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python  # For extra modules
   ```

### Python

For most users, the easiest way to install OpenCV is via pip:

```bash
pip install opencv-python
```

For additional modules (face, text, etc.):

```bash
pip install opencv-contrib-python
```

Verify installation:

```python
import cv2
print(cv2.__version__)
```

<!-- BASICS -->
## Basics

### Reading and Displaying Images

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
img = cv2.imread('image.jpg')

# Check if image loaded successfully
if img is None:
    print("Error: Could not read image")
    exit()

# Display using OpenCV (BGR format)
cv2.imshow('Image', img)
cv2.waitKey(0)  # Wait for any key press
cv2.destroyAllWindows()

# Display using Matplotlib (RGB format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Image')
plt.axis('off')
plt.show()
```

### Writing Images

```python
# Save an image
cv2.imwrite('output.jpg', img)
```

### Reading and Writing Videos

```python
# Read video from file
cap = cv2.VideoCapture('video.mp4')

# Read video from webcam
# cap = cv2.VideoCapture(0)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process frame
    
    # Write the frame
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
```

### Basic Image Operations

```python
# Access pixel values
px = img[100, 100]  # Returns BGR values
blue = img[100, 100, 0]  # Access blue channel

# Modify pixel values
img[100, 100] = [255, 0, 0]  # Set pixel to blue

# Image properties
height, width, channels = img.shape
size = img.size  # Total number of pixels
dtype = img.dtype  # Data type

# Region of Interest (ROI)
roi = img[100:200, 100:200]

# Split and merge channels
b, g, r = cv2.split(img)
img_merged = cv2.merge((b, g, r))

# Copy images properly
img_copy = img.copy()
```

### Drawing Functions

```python
# Create a blank image
blank = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a line
cv2.line(blank, (0, 0), (blank.shape[1], blank.shape[0]), (0, 255, 0), 2)

# Draw a rectangle
cv2.rectangle(blank, (100, 100), (300, 300), (0, 0, 255), 2)  # -1 for filled

# Draw a circle
cv2.circle(blank, (250, 250), 50, (255, 0, 0), -1)  # -1 for filled

# Draw an ellipse
cv2.ellipse(blank, (250, 250), (100, 50), 0, 0, 360, (255, 255, 0), -1)

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank, 'OpenCV', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Draw a polygon
pts = np.array([[100, 50], [200, 300], [400, 200], [300, 100]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(blank, [pts], True, (0, 255, 255), 3)
```

<!-- IMAGE PROCESSING -->
## Image Processing

### Color Spaces

```python
# Convert BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Convert BGR to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Color filtering (HSV example)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)
```

### Image Arithmetic

```python
# Image addition
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
added = cv2.add(img1, img2)  # Saturated addition
weighted = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)  # Weighted addition

# Bitwise operations
bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_xor = cv2.bitwise_xor(img1, img2)
bitwise_not = cv2.bitwise_not(img1)
```

### Thresholding

```python
# Simple thresholding
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# Adaptive thresholding
adaptive_thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
adaptive_thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

# Otsu's thresholding
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Smoothing and Blurring

```python
# Averaging
blur = cv2.blur(img, (5, 5))

# Gaussian blur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Median blur (good for salt-and-pepper noise)
median = cv2.medianBlur(img, 5)

# Bilateral filter (preserves edges)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### Morphological Operations

```python
# Define kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(thresh1, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(thresh1, kernel, iterations=1)

# Opening (Erosion followed by dilation)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by erosion)
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

# Morphological gradient
gradient = cv2.morphologyEx(thresh1, cv2.MORPH_GRADIENT, kernel)

# Top hat
tophat = cv2.morphologyEx(thresh1, cv2.MORPH_TOPHAT, kernel)

# Black hat
blackhat = cv2.morphologyEx(thresh1, cv2.MORPH_BLACKHAT, kernel)
```

### Gradients and Edge Detection

```python
# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
sobel_angle = np.arctan2(sobely, sobelx)

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)
```

### Image Pyramids

```python
# Gaussian Pyramid
lower_res = cv2.pyrDown(img)  # Downsampling
higher_res = cv2.pyrUp(lower_res)  # Upsampling

# Laplacian Pyramid
# Generate Gaussian pyramid
G = img.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Generate Laplacian Pyramid
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1], GE)
    lpA.append(L)
```

### Contours

```python
# Find contours
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

# Contour properties
for cnt in contours:
    # Area
    area = cv2.contourArea(cnt)
    
    # Perimeter
    perimeter = cv2.arcLength(cnt, True)
    
    # Centroid
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(contour_img, center, radius, (0, 255, 0), 2)
    
    # Fit ellipse
    if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(contour_img, ellipse, (0, 255, 0), 2)
    
    # Convex hull
    hull = cv2.convexHull(cnt)
    cv2.drawContours(contour_img, [hull], 0, (0, 0, 255), 2)
```

### Histograms

```python
# Grayscale histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Color histogram
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist_color = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_color, color=col)
    plt.xlim([0, 256])

# Histogram equalization
equ = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl_img = clahe.apply(gray)

# 2D histogram
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist_2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist_2d, interpolation='nearest')
```

<!-- FEATURE DETECTION AND DESCRIPTION -->
## Feature Detection and Description

### Harris Corner Detection

```python
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate to mark the corners
dst = cv2.dilate(dst, None)

# Threshold for optimal value
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red
```

### SIFT (Scale-Invariant Feature Transform)

```python
# Create SIFT object
sift = cv2.SIFT_create()

# Find keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

### SURF (Speeded-Up Robust Features)

```python
# Note: SURF is patented and moved to contrib
# Install opencv-contrib-python to use it
surf = cv2.xfeatures2d.SURF_create(400)  # Threshold (higher -> fewer points)
keypoints, descriptors = surf.detectAndCompute(gray, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, None, (255, 0, 0), 4)
```

### ORB (Oriented FAST and Rotated BRIEF)

```python
# Create ORB object
orb = cv2.ORB_create()

# Find keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
```

### Feature Matching

```python
# FLANN-based matcher for SIFT/SURF
img1 = cv2.imread('object.jpg', 0)
img2 = cv2.imread('scene.jpg', 0)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# Homography (Find object in scene)
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Draw outline
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
```

<!-- VIDEO ANALYSIS -->
## Video Analysis

### Background Subtraction

```python
# Create background subtractor objects
backSub1 = cv2.createBackgroundSubtractorMOG2()
backSub2 = cv2.createBackgroundSubtractorKNN()

# Process video frames
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fgMask1 = backSub1.apply(frame)
    fgMask2 = backSub2.apply(frame)
    
    # Show results
    cv2.imshow('Frame', frame)
    cv2.imshow('MOG2', fgMask1)
    cv2.imshow('KNN', fgMask2)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Optical Flow

```python
# Lucas-Kanade Optical Flow
cap = cv2.VideoCapture('video.mp4')
# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get first frame and find corners
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create mask for drawing
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    
    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', img)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()

# Dense Optical Flow
cap = cv2.VideoCapture('video.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Convert to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Use angle and magnitude to create HSV image
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Dense Optical Flow', bgr)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    prvs = next

cap.release()
cv2.destroyAllWindows()
```

### Object Tracking

```python
# Initialize tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]  # CSRT is generally more accurate

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':  # Requires additional model file
    tracker = cv2.TrackerGOTURN_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

# Read video
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()

# Define initial bounding box
bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker
    ret, bbox = tracker.update(frame)
    
    # Draw bounding box
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Display result
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
