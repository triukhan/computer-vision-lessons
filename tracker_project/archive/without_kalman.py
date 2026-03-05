import cv2
import numpy as np

# ===== Color to be tracked (BLUE) =====
MIN_H_BLUE = 200
MAX_H_BLUE = 300
# =====================================


def main():
    # ---------- Kalman Filter ----------
    state_size = 6   # [x, y, vx, vy, w, h]
    meas_size = 4    # [x, y, w, h]

    kf = cv2.KalmanFilter(state_size, meas_size)

    # State & measurement
    state = np.zeros((state_size, 1), np.float32)
    meas = np.zeros((meas_size, 1), np.float32)

    # Transition matrix A
    kf.transitionMatrix = np.eye(state_size, dtype=np.float32)

    # Measurement matrix H
    kf.measurementMatrix = np.zeros((meas_size, state_size), np.float32)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 4] = 1.0
    kf.measurementMatrix[3, 5] = 1.0

    # Process noise covariance Q
    kf.processNoiseCov = np.diag([
        1e-2,  # x
        1e-2,  # y
        5.0,   # vx
        5.0,   # vy
        1e-2,  # w
        1e-2   # h
    ]).astype(np.float32)

    # Measurement noise covariance R
    kf.measurementNoiseCov = np.eye(meas_size, dtype=np.float32) * 1e-1

    # ---------- Camera ----------
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Webcam not connected")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    found = False
    not_found_count = 0
    ticks = cv2.getTickCount()

    print("Press Q to exit")

    while True:
        prev_ticks = ticks
        ticks = cv2.getTickCount()
        dT = (ticks - prev_ticks) / cv2.getTickFrequency()

        ret, frame = cap.read()
        if not ret:
            break

        res = frame.copy()

        # ===== Prediction =====
        if found:
            kf.transitionMatrix[0, 2] = dT
            kf.transitionMatrix[1, 3] = dT

            state = kf.predict()

            cx, cy, w, h = state[0, 0], state[1, 0], state[4, 0], state[5, 0]

            x = int(cx - w / 2)
            y = int(cy - h / 2)

            cv2.rectangle(res, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
            cv2.circle(res, (int(cx), int(cy)), 2, (255, 0, 0), -1)

        # ===== Image processing =====
        blur = cv2.GaussianBlur(frame, (5, 5), 3)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            (MIN_H_BLUE // 2, 100, 80),
            (MAX_H_BLUE // 2, 255, 255)
        )

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("Threshold", mask)

        # ===== Contours =====
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        balls = []
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = min(w / h, h / w)

            if ratio > 0.75 and w * h >= 400:
                balls.append(cnt)
                boxes.append((x, y, w, h))

        # ===== Detection result =====
        for (x, y, w, h) in boxes:
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(res, (cx, cy), 2, (0, 255, 0), -1)
            cv2.putText(
                res,
                f"({cx},{cy})",
                (cx + 3, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ===== Kalman update =====
        if len(boxes) == 0:
            not_found_count += 1
            if not_found_count >= 100:
                found = False
        else:
            not_found_count = 0
            x, y, w, h = boxes[0]

            meas[:, 0] = [
                x + w / 2,
                y + h / 2,
                w,
                h
            ]

            if not found:
                kf.errorCovPre = np.eye(state_size, dtype=np.float32)
                state[:, 0] = [meas[0, 0], meas[1, 0], 0, 0, meas[2, 0], meas[3, 0]]
                kf.statePost = state
                found = True
            else:
                kf.correct(meas)

        cv2.imshow("Tracking", res)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
