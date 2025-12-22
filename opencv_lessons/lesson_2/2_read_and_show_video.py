import cv2
import os


video_path = os.path.join('opencv_lessons', 'data', 'video.mp4')

video = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = video.read()

    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(40)

video.release()
cv2.destroyAllWindows()
