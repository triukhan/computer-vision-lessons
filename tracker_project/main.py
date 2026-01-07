import cv2
import numpy as np

BOX_W, BOX_H = 30, 30
lk_params = dict(
    winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
tracking = False
points: np.ndarray | None = None
bbox_cx, bbox_cy = 0, 0
VIDEO_PATH = 'istockphoto-608862666-640_adpp_is.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('video')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
kalman.errorCovPost = np.eye(4, dtype=np.float32)


def init_points(gray_frame, cx, cy):
    x_1 = max(0, int(cx - BOX_W // 2))
    y_1 = max(0, int(cy - BOX_H // 2))
    gray_roi = gray_frame[y_1:y_1 + BOX_H, x_1:x_1 + BOX_W]

    pts = cv2.goodFeaturesToTrack(
        gray_roi, maxCorners=80, qualityLevel=0.01, minDistance=7
    )

    if pts is None:
        return None

    pts += np.array([[x_1, y_1]], dtype=np.float32)
    return pts


def mouse_callback(event, x_p, y_p, _, __):
    global tracking, points, bbox_cx, bbox_cy

    if event == cv2.EVENT_LBUTTONDOWN:
        print('fdddd')
        print(x_p, y_p)
        bbox_cx, bbox_cy = x_p, y_p

        kalman.statePost = np.array([
            [x_p],
            [y_p],
            [0],
            [0]
        ], np.float32)
        kalman.statePre = kalman.statePost.copy()

        points = init_points(prev_gray, bbox_cx, bbox_cy)
        tracking = True


cv2.setMouseCallback('video', mouse_callback)
mouse_callback(cv2.EVENT_LBUTTONDOWN, 176, 221, 0, 0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking:
        prediction = kalman.predict()
        predict_x, predict_y = prediction[0, 0], prediction[1, 0]
        measurement_ok = False

        if points is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)
            good_new = new_pts[status.flatten() == 1]
            good_old = points[status.flatten() == 1]

            if len(good_new) >= 8:
                shifts = (good_new - good_old).reshape(-1, 2)
                dx, dy = np.median(shifts, axis=0)
                meas_x, meas_y = bbox_cx + dx, bbox_cy + dy

                kalman.correct(np.array([[meas_x], [meas_y]], np.float32))
                measurement_ok = True
                points = good_new.reshape(-1, 1, 2)

        if not measurement_ok:
            print('predict')
            points = init_points(gray, predict_x, predict_y)

        state = kalman.statePost
        state[2, 0] = np.clip(state[2, 0], -20, 20)  # vx
        state[3, 0] = np.clip(state[3, 0], -20, 20)  # vy
        kalman.statePost = state

        bbox_cx, bbox_cy = state[0, 0], state[1, 0]

        x1 = int(bbox_cx - BOX_W // 2)
        y1 = int(bbox_cy - BOX_H // 2)
        x2 = x1 + BOX_W
        y2 = y1 + BOX_H

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    prev_gray = gray.copy()
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
