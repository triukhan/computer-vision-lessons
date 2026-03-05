import cv2
import numpy as np

BOX_W, BOX_H = 30, 30
lk_params = dict(
    winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
tracking = False
points: np.ndarray | None = None
bbox_cx, bbox_cy = 0, 0
VIDEO_PATH = 'istockphoto-1387330605-640_adpp_is.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


kalman = cv2.KalmanFilter(6, 4)
kalman.transitionMatrix = np.eye(6, dtype=np.float32)
kalman.measurementMatrix = np.zeros((4, 6), np.float32)
kalman.measurementMatrix[0, 0] = 1  # x
kalman.measurementMatrix[1, 1] = 1  # y
kalman.measurementMatrix[2, 4] = 1  # w
kalman.measurementMatrix[3, 5] = 1  # h
lk_cx, lk_cy = None, None


kalman.processNoiseCov = np.diag([
    1e-2,  # x
    1e-2,  # y
    5.0,   # vx
    5.0,   # vy
    1e-3,  # w
    1e-3,  # h
]).astype(np.float32)
kalman.measurementNoiseCov = np.diag([
    1e-1,  # x
    1e-1,  # y
    5.0,   # w
    5.0,   # h
]).astype(np.float32)
kalman.errorCovPost = np.eye(6, dtype=np.float32)

state = np.zeros((6, 1), np.float32)
meas = np.zeros((4, 1), np.float32)


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
    print('aaaa', pts)
    return pts


def mouse_callback(event, x_p, y_p, _, __):
    global tracking, points, bbox_cx, bbox_cy, lk_cx, lk_cy

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x_p, y_p)
        bbox_cx, bbox_cy = x_p, y_p
        lk_cx, lk_cy = bbox_cx, bbox_cy

        kalman.statePost = np.array([
            [x_p],  # x
            [y_p],  # y
            [0.0],  # vx
            [0.0],  # vy
            [BOX_W],  # w
            [BOX_H],  # h
        ], dtype=np.float32)
        kalman.statePre = kalman.statePost.copy()

        points = init_points(prev_gray, bbox_cx, bbox_cy)
        tracking = True


cv2.namedWindow('video', cv2.WINDOW_NORMAL)
mouse_callback(cv2.EVENT_LBUTTONDOWN, 422, 271, 0, 0)
ticks = cv2.getTickCount()


while True:
    prev_ticks = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - prev_ticks) / cv2.getTickFrequency()  # iteration time

    ret, frame = cap.read()
    if not ret:
        break

    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking:
        kalman.transitionMatrix[0, 2] = dT
        kalman.transitionMatrix[1, 3] = dT

        state = kalman.predict()

        cx, cy, w, h = state[0, 0], state[1, 0], state[4, 0], state[5, 0]
        measurement_ok = False

        if points is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)
            good_new = new_pts[status.flatten() == 1]
            good_old = points[status.flatten() == 1]

            if len(good_new) >= 8:
                shifts = (good_new - good_old).reshape(-1, 2)
                dx, dy = np.median(shifts, axis=0)

                lk_cx, lk_cy = bbox_cx, bbox_cy 
                meas_x = lk_cx + dx
                meas_y = lk_cy + dy
                lk_cx, lk_cy = meas_x, meas_y

                meas = np.array([
                    [meas_x],
                    [meas_y],
                    [BOX_W],
                    [BOX_H],
                ], dtype=np.float32)
                kalman.correct(meas)
                measurement_ok = True
                points = good_new.reshape(-1, 1, 2)

        if not measurement_ok:
            print('predict')
            points = init_points(gray, cx, cy)

        state = kalman.statePost
        state[2, 0] = np.clip(state[2, 0], -20, 20)  # vx
        state[3, 0] = np.clip(state[3, 0], -20, 20)  # vy
        kalman.statePost = state

        bbox_cx, bbox_cy = state[0, 0], state[1, 0]

        cx, cy = int(state[0, 0]), int(state[1, 0])
        w, h = int(state[4, 0]), int(state[5, 0])

        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        print((x1, y1), (x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    prev_gray = gray.copy()
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.setMouseCallback('video', mouse_callback)

    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
