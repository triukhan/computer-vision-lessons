import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from siam_tracker.model import BaselineEmbeddingNet, SiameseTracker
import torchvision.models as models

cap = cv2.VideoCapture('/home/danylo/GIT/computer-vision-lessons/tracker_project/helicopter.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

bbox = None
need_init = False


def crop(frame, cbbox, size, context_factor=0.5):
    cx, cy, w, h = cbbox
    context = context_factor * (w + h)
    crop_size = int(np.sqrt((w + context) * (h + context)))

    x1 = int(cx - crop_size / 2)
    y1 = int(cy - crop_size / 2)
    x2 = int(cx + crop_size / 2)
    y2 = int(cy + crop_size / 2)

    left = max(0, -x1)
    top = max(0, -y1)
    right = max(0, x2 - frame.shape[1])
    bottom = max(0, y2 - frame.shape[0])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    cropped = frame[y1:y2, x1:x2]

    if top or bottom or left or right:
        cropped = cv2.copyMakeBorder(
            cropped, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0,0,0)
        )

    cropped = cv2.resize(cropped, (size, size))

    return cropped


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor

def update_bbox(prev_bbox, pos, response_size=8, stride=8):
    cx, cy, w, h = prev_bbox
    row = pos // response_size
    col = pos % response_size
    center = response_size // 2

    disp_x = (col - center) * stride
    disp_y = (row - center) * stride

    search_size = 255
    cx_s, cy_s, w_s, h_s = prev_bbox
    context = 0.8 * (w_s + h_s)
    crop_size = np.sqrt((w_s + context) * (h_s + context))
    scale = crop_size / search_size

    new_cx = cx + disp_x * scale
    new_cy = cy + disp_y * scale

    return new_cx, new_cy, w, h

def cross_correlation(z, x):
    b, c, h, w = z.shape
    z_kernel = z.view(c, 1, h, w)
    response = F.conv2d(x, z_kernel, groups=c)
    response = response.mean(dim=1, keepdim=True)

    return response


class SiamTracker:
    def __init__(self, model):
        self.model = model
        checkpoint = torch.load("/home/danylo/GIT/computer-vision-lessons/BaselinePretrained.pth.tar", map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.template_feature = None
        self.template = None

    def init(self, frame, ibbox):
        template = crop(frame, ibbox, size=127, context_factor=0.6)
        template = preprocess(template)
        self.template = template

    def track(self, frame, prev_bbox):
        search = crop(frame, prev_bbox, size=255, context_factor=0.8)
        search = preprocess(search)

        with torch.no_grad():
            response = self.model(self.template, search)  # 🔥 ВСЕ тут

        response_np = response.squeeze().cpu().numpy()

        # нормалізація
        response_np = response_np - response_np.mean()
        response_np = response_np / (response_np.std() + 1e-8)

        # Hanning window
        h, w = response_np.shape
        hanning = np.outer(np.hanning(h), np.hanning(w))
        window_influence = 0.40
        response_np = (1 - window_influence) * response_np + window_influence * hanning

        # максимум
        pos = np.argmax(response_np)

        # debug
        debug_norm = response_np - response_np.min()
        debug_norm = debug_norm / (debug_norm.max() + 1e-8)
        debug = cv2.resize(debug_norm, (200, 200), interpolation=cv2.INTER_NEAREST)
        debug = (debug * 255).astype(np.uint8)
        debug_color = cv2.applyColorMap(debug, cv2.COLORMAP_JET)
        cv2.imshow('response map', debug_color)

        return update_bbox(prev_bbox, pos, response_size=h, stride=8)


def on_mouse(event, x, y, _, __):
    global bbox, need_init

    if event == cv2.EVENT_LBUTTONDOWN:
        w, h = 60, 60
        bbox = (x, y, w, h)
        print(bbox)

        need_init = True


def track_object(video, stop=False):
    global need_init, bbox
    counter, kf, points, prev_gray, skip_first = 0, None, None, None, False

    tracker = SiamTracker(SiameseTracker())
    cv2.namedWindow('tracking with siam')
    cv2.setMouseCallback('tracking with siam', on_mouse)

    while True:
        counter += 1
        ret, frame = video.read()
        if not ret:
            break

        if need_init:
            tracker.init(frame, bbox)
            need_init = False
            skip_first = True

        if skip_first:
            cx, cy, w, h = bbox
            cv2.rectangle(frame, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (0, 0, 255), 1)
            skip_first = False
        elif bbox:
            bbox = tracker.track(frame, bbox)
            cx, cy, w, h = bbox
            cv2.rectangle(frame, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (0, 0, 255), 1)

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


on_mouse(cv2.EVENT_LBUTTONDOWN, 166, 221, None, None)
track_object(video=cap, stop=False)
