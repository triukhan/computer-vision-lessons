import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')
image = cv2.imread(image_path)

resized_image = cv2.resize(image, (300, 500))

print(image.shape)
print(resized_image.shape)

cv2.imshow('img', resized_image)
cv2.waitKey(0)
