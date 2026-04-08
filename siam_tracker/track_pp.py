from pathlib import Path

import cv2

from siam_tracker.model_builder import ModelBuilder
from siam_tracker.tracker import NanoTracker

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def track_object(video_path: Path, stop=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = NanoTracker(ModelBuilder())
    cv2.namedWindow('tracking with siam')
    cv2.setMouseCallback('tracking with siam', tracker.on_mouse)
    # tracker.on_mouse(cv2.EVENT_LBUTTONDOWN, 166, 221, None, None)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if tracker.need_init:
            tracker.init(frame, tracker.bbox)
            tracker.need_init = False

        if tracker.center_pos is not None:
            tracker.bbox = tracker.track(frame)['bbox']
            cx, cy, w, h = tracker.bbox
            cv2.rectangle(
                frame, (int(cx - w // 2), int(cy - h // 2)), (int(cx + w // 2), int(cy + h // 2)), (0, 0, 255), 1
            )

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'tracker_project' / 'helicopter.mp4'
track_object(vid, stop=True)
