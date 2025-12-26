import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')
image = cv2.imread(image_path)

print(image.shape)

cropped_image = image[230:450, 200:400]

print(cropped_image.shape)

cv2.imshow('img', cropped_image)
cv2.waitKey(0)
