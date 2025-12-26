import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')
image = cv2.imread(image_path)

k_size = 7
image_blur = cv2.blur(image, (k_size, k_size))
image_gaussian_blur = cv2.GaussianBlur(image, (k_size, k_size), 3)
image_median_blur = cv2.medianBlur(image, k_size)

cv2.imshow('image_blur', image_blur)
cv2.imshow('image_gaussian_blur', image_gaussian_blur)
cv2.imshow('image_median_blur', image_median_blur)  # removing noise

cv2.waitKey(0)
