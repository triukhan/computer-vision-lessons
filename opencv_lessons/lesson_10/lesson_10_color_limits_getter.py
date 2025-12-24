import numpy
import cv2


def get_limits(color):
    color = numpy.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = numpy.array(lower_limit, dtype=numpy.uint8)
    upper_limit = numpy.array(upper_limit, dtype=numpy.uint8)

    return lower_limit, upper_limit
