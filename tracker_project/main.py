import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

bbox = None
need_init = False


def create_kalman():
    kalman_f = cv2.KalmanFilter(8, 4)

    kalman_f.transitionMatrix = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ], np.float32)

    kalman_f.measurementMatrix = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0]
    ], np.float32)

    kalman_f.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
    kalman_f.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    return kalman_f


def on_mouse(event, x, y, _, __):
    global bbox, need_init

    if event == cv2.EVENT_LBUTTONDOWN:
        w, h = 60, 60
        bbox = (x - w//2, y - h//2, w, h)
        print(bbox)

        need_init = True


def detect_features(gray, bbox_to_detect):
    margin = 0.2

    x, y, w, h = map(int, bbox_to_detect)

    x2 = int(x + w * margin)
    y2 = int(y + h * margin)
    w2 = int(w * (1 - 2 * margin))
    h2 = int(h * (1 - 2 * margin))

    mask = np.zeros_like(gray)
    mask[y2:y2 + h2, x2:x2 + w2] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask
    )

    return pts


def bbox_from_corners(corners):
    x = np.min(corners[:,0])
    y = np.min(corners[:,1])

    w = np.max(corners[:,0]) - x
    h = np.max(corners[:,1]) - y

    return x, y, w, h


def track_object_with_kalman(video, stop=False):
    global need_init, bbox
    counter, kf, points, prev_gray = 0, None, None, None

    cv2.namedWindow('tracking with kalman')
    cv2.setMouseCallback('tracking with kalman', on_mouse)

    while True:
        counter += 1
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if need_init:
            kf = create_kalman()

            points = detect_features(gray, bbox)

            x, y, w, h = bbox
            bbox_center_x, bbox_center_y = x + w/2, y + h/2  # kalman works with centers

            # the best estimate after correction
            kf.statePost = np.array([
                [bbox_center_x], [bbox_center_y], [w], [h],
                [0], [0], [0], [0]  # zeros - we don`t have these values so far, cuz this is the first frame
            ], np.float32)

            kf.statePre = kf.statePost.copy()  # just for correct initializing. by default statePre has zeros
            kf.errorCovPost = np.eye(8, dtype=np.float32)  # set a shape of uncertainty

            prev_gray = gray.copy()
            need_init = False

        if points is not None and prev_gray is not None:
            # for the first frame filter will not affect tracker
            prediction = kf.predict()  # predicts with statePre, statePost and transitionMatrix
            px, py, pw, ph = prediction[:4].flatten()

            pred_x = int(px - pw/2)
            pred_y = int(py - ph/2)

            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3
            )

            if new_pts is not None:
                good_new = new_pts[status == 1]
                good_old = points[status == 1]

                if len(good_new) >= 6:
                    M, inliers = cv2.estimateAffinePartial2D(
                        good_old,
                        good_new,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3
                    )

                    if M is not None:
                        inliers = inliers.ravel().astype(bool)

                        good_new = good_new[inliers]
                        good_old = good_old[inliers]

                        # median flow
                        flow = good_new - good_old
                        flow = flow.reshape(-1, 2)
                        dx, dy = np.median(flow, axis=0)

                        # center_old = np.median(good_old, axis=0)
                        center_new = np.median(good_new, axis=0)

                        # dist_old = np.linalg.norm(good_old - center_old, axis=1)
                        # dist_new = np.linalg.norm(good_new - center_new, axis=1)

                        # scale = np.median(dist_new / (dist_old + 1e-6))
                        # scale = np.clip(scale, 0.9, 1.1)

                        # move bbox
                        x, y, w, h = bbox
                        x += dx
                        y += dy
                        # w *= scale
                        # h *= scale

                        # center bbox by features
                        cx, cy = center_new
                        x = cx - w / 2
                        y = cy - h / 2

                        bbox = (x, y, w, h)

                        # corners = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=np.float32)
                        # new_corners = cv2.transform(np.array([corners]), M)[0]
                        # x, y, w, h = bbox_from_corners(new_corners)
                        # bbox = (x, y, w, h)

                        # kalman correction
                        cx, cy = x + w / 2, y + h / 2
                        measurement = np.array([[cx], [cy], [w], [h]], np.float32)

                        kf.correct(measurement)

                        fx, fy, fw, fh = kf.statePost[:4].flatten()

                        # draw
                        draw_x, draw_y = int(fx - fw / 2), int(fy - fh / 2)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
                        cv2.rectangle(frame, (pred_x, pred_y), (int(pred_x + pw), int(pred_y + ph)), (255, 0, 0), 1)
                        cv2.rectangle(frame, (draw_x, draw_y), (int(draw_x + fw), int(draw_y + fh)), (0, 255, 0), 1)

                        for p in good_new:
                            x, y = p.ravel()
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                        points = good_new.reshape(-1, 1, 2)

                        if len(points) < 20:
                            points = detect_features(gray, bbox)
            prev_gray = gray.copy()

        cv2.imshow('tracking with kalman', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            # stop = False

    video.release()
    cv2.destroyAllWindows()


track_object_with_kalman(video=cap, stop=False)
