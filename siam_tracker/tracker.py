from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from siam_tracker.model import BaselineEmbeddingNet, SiameseTracker
import torchvision.models as models


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'BaselinePretrained.pth.tar'
H, W = 60, 60


def crop(frame, bbox, size: int, context_factor: float = 0.5):
    cx, cy, w, h = bbox

    # expand bbox with contex factor
    context = context_factor * (w + h)
    crop_size = int(np.sqrt((w + context) * (h + context)))  # sqrt saves proportions
    x1, y1 = int(cx - crop_size / 2), int(cy - crop_size / 2)
    x2, y2 = int(cx + crop_size / 2), int(cy + crop_size / 2)

    # if edge of the frame - add black pixels instead
    left, right = max(0, -x1), max(0, x2 - frame.shape[1])
    top, bottom = max(0, -y1), max(0, y2 - frame.shape[0])

    x1, y1 = max(0, x1), max(0, y1)  # clamping after expanding
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    cropped = frame[y1:y2, x1:x2]

    if top or bottom or left or right:
        cropped = cv2.copyMakeBorder(
            cropped, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0)
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
    row = pos // response_size
    col = pos % response_size
    center = response_size // 2

    disp_x = (col - center) * stride
    disp_y = (row - center) * stride

    search_size = 255
    prev_cx, prev_cy, prev_w, prev_h = prev_bbox
    context = 0.8 * (prev_w + prev_h)
    crop_size = np.sqrt((prev_w + context) * (prev_h + context))
    scale = crop_size / search_size

    new_cx = prev_cx + disp_x * scale
    new_cy = prev_cy + disp_y * scale

    # smoothing
    alpha = 0.5
    new_cx = alpha * new_cx + (1 - alpha) * prev_cx
    new_cy = alpha * new_cy + (1 - alpha) * prev_cy

    return new_cx, new_cy, 60, 60


class SiamTracker:
    def __init__(self, model):
        self.model = model
        self.template_feature = None
        self.original_template_feature = None
        self.bbox, self.need_init = None, False

        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def init(self, frame, bbox):
        template = crop(frame, bbox, size=127, context_factor=0.6)
        template = preprocess(template)
        with torch.no_grad():
            self.template_feature = self.model.embedding_net(template)
        self.original_template_feature = self.template_feature.clone()

    def track(self, frame, prev_bbox):
        search = crop(frame, prev_bbox, size=255, context_factor=0.8)
        search = preprocess(search)

        with torch.no_grad():
            search_feature = self.model.embedding_net(search)  # todo: use forward() instead
            response = self.model.match_corr(self.template_feature, search_feature)

        # normalization
        response_np = response.squeeze().cpu().numpy()
        response_np = response_np - response_np.mean()
        response_np = response_np / (response_np.std() + 1e-8)

        # a bit correct to center
        h, w = response_np.shape
        hanning = np.outer(np.hanning(h), np.hanning(w))
        window_influence = 0.4
        response_np = (1 - window_influence) * response_np + window_influence * hanning

        pos = np.argmax(response_np)

        debug_norm = response_np - response_np.min()
        debug_norm = debug_norm / (debug_norm.max() + 1e-8)
        debug = cv2.resize(debug_norm, (200, 200), interpolation=cv2.INTER_NEAREST)
        debug = (debug * 255).astype(np.uint8)
        debug_color = cv2.applyColorMap(debug, cv2.COLORMAP_JET)
        cv2.imshow('response map', debug_color)

        new_bbox = update_bbox(prev_bbox, pos, response_size=h, stride=8)

        response_max = response_np.max()
        confidence_threshold = 2

        if response_max > confidence_threshold:
            print(response_max)
            new_template = preprocess(crop(frame, new_bbox, size=127, context_factor=0.6))
            with torch.no_grad():
                new_feature = self.model.embedding_net(new_template)
            alpha = 0.1
            self.template_feature = alpha * new_feature + (1 - alpha) * self.template_feature

        gamma = 0.8
        self.template_feature = gamma * self.template_feature + (1 - gamma) * self.original_template_feature

        return new_bbox


    def on_mouse(self, event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (x, y, w, h)
            self.need_init = True
