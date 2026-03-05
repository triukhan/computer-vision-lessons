import cv2
import numpy as np

cap = cv2.VideoCapture('5004429-uhd_3840_2160_30fps.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 110)

tracker = None
kf = None
bbox = None
need_init = False


def create_kalman():
    kf = cv2.KalmanFilter(8, 4)

    kf.transitionMatrix = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ], np.float32)

    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0]
    ], np.float32)

    kf.processNoiseCov = np.diag([
        1e-2, 1e-2, 1e-2, 1e-2,
        5e-1, 5e-1, 1e-1, 1e-1
    ]).astype(np.float32)

    kf.measurementNoiseCov = np.diag([1, 1, 10, 10]).astype(np.float32)

    return kf


def on_mouse(event, x, y, flags, param):
    global bbox, need_init

    if event == cv2.EVENT_LBUTTONDOWN:

        w, h = 60, 60
        bbox = (x - w//2, y - h//2, w, h)
        print(bbox)

        need_init = True


def track_object_with_kalman(video, stop=False):
    global tracker, kf, need_init, bbox

    cv2.namedWindow('tracking with kalman')
    cv2.setMouseCallback('tracking with kalman', on_mouse)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if need_init:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            kf = create_kalman()

            x, y, w, h = bbox
            bbox_center_x, bbox_center_y = x + w/2, y + h/2  # kalman works with centers

            # the best estimate after correction
            kf.statePost = np.array([
                [bbox_center_x], [bbox_center_y], [w], [h],
                [0], [0], [0], [0]  # zeros - we don`t have these values so far, cuz this is the first frame
            ], np.float32)

            kf.statePre = kf.statePost.copy()  # just for correct initializing. by default statePre has zeros
            kf.errorCovPost = np.eye(8, dtype=np.float32)  # set a shape of uncertainty

            need_init = False

        if tracker is not None:
            # for the first frame filter will not affect tracker
            pred = kf.predict()  # predicts with statePre, statePost and transitionMatrix
            px, py, pw, ph = pred[:4].flatten()

            pred_x = int(px - pw/2)
            pred_y = int(py - ph/2)

            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = bbox
                bbox_center_x, bbox_center_y = x + w/2, y + h/2
                measurement = np.array([[bbox_center_x], [bbox_center_y], [w], [h]], np.float32)

                kf.correct(measurement)  # here figures out new statePost & errorCovPost

                px, py, pw, ph = kf.statePost[:4].flatten()

                draw_x, draw_y = int(px - pw/2), int(py - ph/2)
                cv2.rectangle(frame, (draw_x, draw_y), (int(draw_x + pw), int(draw_y + ph)), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (pred_x, pred_y), (int(pred_x + pw), int(pred_y + ph)), (0, 0, 255), 2)

        cv2.imshow('tracking with kalman', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            stop = False

    video.release()
    cv2.destroyAllWindows()


def track_object(tracker, cap, bbox):
    ret, frame = cap.read()
    tracker.init(frame, bbox)

    x, y, w, h = bbox
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imshow('tracking', frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, bbox = tracker.update(frame)

        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# tr = cv2.TrackerCSRT_create()
# track_object(tr, cap, bbox=(1222, 1190, 60, 60))

bbox = (1222, 1190, 60, 60)
need_init = True

track_object_with_kalman(video=cap, stop=True)