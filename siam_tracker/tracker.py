import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.models as models
from siam_tracker.model import Backbone

cap = cv2.VideoCapture('/Users/danylo/Documents/computer-vision-lessons/tracker_project/helicopter.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

bbox = None
need_init = False


def crop(frame, bbox, size=255):
    cx, cy, w, h = bbox

    context = 0.5 * (w + h)
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


def update_bbox(prev_bbox, pos, response_size=17, stride=8):
    cx, cy, w, h = prev_bbox

    row = pos // response_size
    col = pos % response_size

    disp_x = (col - response_size // 2) * stride
    disp_y = (row - response_size // 2) * stride

    new_cx = cx + disp_x
    new_cy = cy + disp_y

    return (new_cx, new_cy, w, h)


class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.alexnet(pretrained=True).features

    def forward(self, template, search):
        z = self.backbone(template)
        x = self.backbone(search)
        response = cross_correlation(z, x)

        return response


def cross_correlation(template, search):
    batch = search.size(0)

    out = []
    for i in range(batch):
        out.append(F.conv2d(search[i:i+1], template[i:i+1]))

    return torch.cat(out)


class SiamTracker:
    def __init__(self, model):
        self.model = model
        self.template_feature = None

    def init(self, frame, bbox):
        template = crop(frame, bbox, size=127)
        template = preprocess(template)

        with torch.no_grad():
            self.template_feature = self.model.backbone(template)

    def track(self, frame, prev_bbox):
        search = crop(frame, prev_bbox, size=255)
        search = preprocess(search)

        with torch.no_grad():

            search_feature = self.model.backbone(search)

            response = cross_correlation(
                self.template_feature,
                search_feature
            )

        response = response.squeeze().cpu().numpy()
        pos = response.view(-1).argmax().item()

        return update_bbox(prev_bbox, pos)


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
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            skip_first = False
        elif bbox:
            bbox = tracker.track(frame, bbox)
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


track_object(video=cap, stop=True)
