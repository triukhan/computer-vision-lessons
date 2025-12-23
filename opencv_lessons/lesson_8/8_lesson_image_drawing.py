import os
import cv2
import numpy

image_path = os.path.join('opencv_lessons', 'data', 'img.png')
image = cv2.imread(image_path)
print(image.shape)

cv2.line(image, (100, 150), (200, 300), (0, 255, 0), 3)
cv2.rectangle(image, (100, 400), (400, 500), (255, 0, 0), -1)
cv2.circle(image, (300, 650), 50, (0, 0, 255), 10)
cv2.putText(image, 'gnom is learning smth', (50, 100), 2, 1, (200, 100, 100), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
