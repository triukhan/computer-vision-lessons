import os
import cv2


image_path = os.path.join('opencv_lessons', 'data', 'img_threshold.png')
image = cv2.imread(image_path)

image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_grey, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 50:
        # cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
