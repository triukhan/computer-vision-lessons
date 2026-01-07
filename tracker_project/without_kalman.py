import cv2
import numpy as np

BOX_W, BOX_H = 30, 30
click_point: tuple | None = None
tracking = False
points = None
bbox_cx, bbox_cy = 0, 0
VIDEO_PATH = 'istockphoto-608862666-640_adpp_is.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)

ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError('cannot read video')

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


def mouse_callback(event, x_p, y_p, _, __):
    global click_point, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x_p, y_p)
        click_point = (x_p, y_p)
        tracking = False


cv2.namedWindow('video')
cv2.setMouseCallback('video', mouse_callback)

lk_params = dict(
    winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

mouse_callback(cv2.EVENT_LBUTTONDOWN, 176, 221, 0, 0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if click_point and not tracking:
        x, y = click_point
        bbox_cx, bbox_cy = x, y

        x1 = x - BOX_W // 2
        y1 = y - BOX_H // 2
        roi = prev_gray[y1:y1 + BOX_H, x1:x1 + BOX_W]

        points = cv2.goodFeaturesToTrack(roi, maxCorners=80, qualityLevel=0.01, minDistance=7)

        points += np.array([[x1, y1]], dtype=np.float32)
        tracking = True
        click_point = None

    if tracking and points is not None:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)

        good_new = new_points[status.flatten() == 1]
        good_old = points[status.flatten() == 1]

        if len(good_new) >= 8:
            shifts = (good_new - good_old).reshape(-1, 2)
            dx, dy = np.median(shifts, axis=0)

            bbox_cx += dx
            bbox_cy += dy

            points = good_new.reshape(-1, 1, 2)

            x1 = int(bbox_cx - BOX_W // 2)
            y1 = int(bbox_cy - BOX_H // 2)
            x2 = x1 + BOX_W
            y2 = y1 + BOX_H

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            tracking = False
            points = None

    prev_gray = gray.copy()
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()