import cv2
from opencv_lessons.lesson_10.lesson_10_color_limits_getter import get_limits
from PIL import Image


cap = cv2.VideoCapture(1)
color = [255, 0, 255]

while True:
    ret, frame = cap.read()
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color)
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
