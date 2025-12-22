import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img.png')
image = cv2.imread(image_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 90, 255, cv2.THRESH_BINARY)
blured_thresh = cv2.blur(thresh, (2, 2))

cv2.imshow('image_gray', image_gray)
cv2.imshow('thresh', thresh)
cv2.imshow('blured_thresh', blured_thresh)
cv2.waitKey(0)
