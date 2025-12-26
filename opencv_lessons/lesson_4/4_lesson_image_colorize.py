import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('img_rgb', image_rgb)
cv2.imshow('image_gray', image_gray)
cv2.imshow('img', image)
cv2.waitKey(0)
