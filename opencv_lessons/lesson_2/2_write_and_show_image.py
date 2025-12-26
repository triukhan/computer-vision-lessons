import cv2
import os


image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')

image = cv2.imread(image_path)

cv2.imwrite(os.path.join('opencv_lessons', 'data', 'img_out.png'), image)

cv2.imshow('image', image)
cv2.waitKey(0)
