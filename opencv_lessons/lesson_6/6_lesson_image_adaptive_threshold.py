import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'adaptive_thresholding.png')
image = cv2.imread(image_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
blured_thresh = cv2.blur(thresh, (2, 2))

cv2.imshow('blured_thresh', blured_thresh)
cv2.waitKey(0)
