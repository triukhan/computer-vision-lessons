import os
import cv2
import numpy

image_path = os.path.join('opencv_lessons', 'data', 'img.jpg')
image = cv2.imread(image_path)

image_edge = cv2.Canny(image, 50, 150)
image_edge_dilate = cv2.dilate(image_edge, numpy.ones((2, 2), dtype=numpy.int8))
image_edge_erode = cv2.erode(image_edge_dilate, numpy.ones((2, 2), dtype=numpy.int8))

cv2.imshow('image_edge', image_edge)
cv2.imshow('image_edge_dilate', image_edge_dilate)
cv2.imshow('image_edge_erode', image_edge_erode)
cv2.waitKey(0)
