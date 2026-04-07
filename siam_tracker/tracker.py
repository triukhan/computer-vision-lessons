from pathlib import Path

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'BaselinePretrained.pth.tar'
H, W = 60, 60


def get_subwindow(frame, bbox, size: int, context_factor: float = 0.5):
    cx, cy, w, h = bbox

    # expand bbox with contex factor
    context = context_factor * (w + h)
    crop_size = int(np.sqrt((w + context) * (h + context)))  # sqrt saves proportions
    channel_average = np.mean(frame, axis=(0, 1))

    x1, y1 = int(cx - crop_size / 2), int(cy - crop_size / 2)
    x2, y2 = int(cx + crop_size / 2), int(cy + crop_size / 2)

    # if edge of the frame - add black pixels instead
    left, right = max(0, -x1), max(0, x2 - frame.shape[1])
    top, bottom = max(0, -y1), max(0, y2 - frame.shape[0])

    x1, y1 = max(0, x1), max(0, y1)  # clamping after expanding
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    r, c, k = frame.shape
    if any([top, bottom, left, right]):
        pad_shape = (r + top + bottom, c + left + right, k)
        te_image = np.zeros(pad_shape, np.uint8)
        te_image[top:top + r, left:left + c, :] = frame
        if top:
            te_image[0:top, left:left + c, :] = channel_average
        if bottom:
            te_image[r + top:, left:left + c, :] = channel_average
        if left:
            te_image[:, 0:left, :] = channel_average
        if right:
            te_image[:, c + left:, :] = channel_average
        image_patch = te_image[int(y1):int(y2), int(x1):int(x2), :]
    else:
        image_patch = frame[int(y1):int(y2), int(x1):int(x2), :]

    image_patch = cv2.resize(image_patch, (size, size))

    return image_patch


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
        # self.template_feature = None
        # self.original_template_feature = None
        self.in_track, self.need_init = False, False
        self.points = self.generate_points(stride=16, size=15)

        # checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        # self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def init(self, frame, bbox):
        template = get_subwindow(frame, bbox, size=127, context_factor=0.6)
        template_tensor = preprocess(template)
        with torch.no_grad():
        #     self.template_feature = self.model.init(template)
            self.model.init(template_tensor)
            self.in_track = True
        # self.original_template_feature = self.template_feature.clone()

    def track(self, frame, prev_bbox):
        search = get_subwindow(frame, prev_bbox, size=255, context_factor=0.8)
        search_tensor = preprocess(search)

        with torch.no_grad():
            # search_feature = self.model.embedding_net(search)  # todo: use forward() instead
            # response = self.model.match_corr(self.template_feature, search_feature)
            response = self.model.track(search_tensor)

        loc = response['loc']

        pred_bbox = self._convert_bbox(loc, self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)

        # score
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * self.cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]

        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr

        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        new_bbox = [cx - width / 2, cy - height / 2, width, height]

        # # normalization
        # response = response['loc']
        # response_np = response.squeeze().cpu().numpy()
        # response_np = response_np - response_np.mean()
        # response_np = response_np / (response_np.std() + 1e-8)
        #
        # # a bit correct to center
        # h, w = response_np.shape
        # hanning = np.outer(np.hanning(h), np.hanning(w))
        # window_influence = 0.4
        # response_np = (1 - window_influence) * response_np + window_influence * hanning
        #
        # pos = np.argmax(response_np)
        #
        # debug_norm = response_np - response_np.min()
        # debug_norm = debug_norm / (debug_norm.max() + 1e-8)
        # debug = cv2.resize(debug_norm, (200, 200), interpolation=cv2.INTER_NEAREST)
        # debug = (debug * 255).astype(np.uint8)
        # debug_color = cv2.applyColorMap(debug, cv2.COLORMAP_JET)
        # cv2.imshow('response map', debug_color)
        #
        # new_bbox = update_bbox(prev_bbox, pos, response_size=h, stride=8)
        #
        # response_max = response_np.max()
        # confidence_threshold = 2
        #
        # if response_max > confidence_threshold:
        #     new_template = preprocess(get_subwindow(frame, new_bbox, size=127, context_factor=0.6))
        #     with torch.no_grad():
        #         new_feature = self.model.embedding_net(new_template)
        #     alpha = 0.1
        #     self.template_feature = alpha * new_feature + (1 - alpha) * self.template_feature
        #
        # gamma = 0.8
        # self.template_feature = gamma * self.template_feature + (1 - gamma) * self.original_template_feature

        return new_bbox

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]  # x1
        delta[1, :] = point[:, 1] - delta[1, :]  # y1
        delta[2, :] = point[:, 0] + delta[2, :]  # x2
        delta[3, :] = point[:, 1] + delta[3, :]  # y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta


    def on_mouse(self, event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (x, y, w, h)
            self.need_init = True


from collections import namedtuple


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
Center = namedtuple('Center', 'x y w h')

def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h
