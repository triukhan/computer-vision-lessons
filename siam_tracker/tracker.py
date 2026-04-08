from pathlib import Path

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'mobilenetv3_small_1.0.pth'
H, W = 60, 60


# def get_subwindow_tracking(frame, target_pos, model_size: int, original_size: int, channel_avg):
#     c = (original_size + 1) / 2
#
#     context_xmin = round(target_pos[0] - c)
#     context_xmax = context_xmin + original_size - 1
#     context_ymin = round(target_pos[1] - c)
#     context_ymax = context_ymin + original_size - 1
#
#     left_pad = int(max(0, -context_xmin))
#     top_pad = int(max(0, -context_ymin))
#     right_pad = int(max(0, context_xmax - frame.shape[1] + 1))
#     bottom_pad = int(max(0, context_ymax - frame.shape[0] + 1))
#
#     # if edge of the frame - add black pixels instead
#     context_xmin += left_pad
#     context_xmax += left_pad
#     context_ymin += top_pad
#     context_ymax += top_pad
#
#     if any([left_pad, top_pad, right_pad, bottom_pad]):
#         te_im = cv2.copyMakeBorder(
#             frame, top=top_pad, bottom=bottom_pad, left=left_pad, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=channel_avg
#         )
#         im_path_original = te_im[
#             context_ymin:context_ymax + 1,
#             context_xmin:context_xmax + 1
#         ]
#     else:
#         im_path_original = frame[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1]
#
#     im_path = cv2.resize(im_path_original, (model_size, model_size))
#
#     return im_path

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
        int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
        int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    # if cfg.CUDA:
    #     im_patch = im_patch.cuda()
    return im_patch


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


class NanoTracker:
    def __init__(self, model):
        self.size = None
        self.center_pos = None
        self.channel_average = None
        self.need_init = False

        self.context_amount = 0.5
        self.score_size = 15
        self.stride = 16

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()

        self.points = self.generate_points(self.stride, self.score_size)
        self.model = model
        self.model.eval()

    @staticmethod
    def generate_points(stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    @staticmethod
    def _convert_bbox(delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :] #x1
        delta[1, :] = point[:, 1] - delta[1, :] #y1
        delta[2, :] = point[:, 0] + delta[2, :] #x2
        delta[3, :] = point[:, 1] + delta[3, :] #y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    @staticmethod
    def _bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, frame, bbox):
        self.center_pos = np.array([bbox[0], bbox[1]])
        self.size = np.array([bbox[2], bbox[3]])

        w = bbox[2]
        h = bbox[3]

        context = self.context_amount * (w + h)
        crop_size = int(np.sqrt((w + context) * (h + context)))  # sqrt saves proportions
        self.channel_average = np.mean(frame, axis=(0, 1))

        z_crop = get_subwindow_tracking(frame, self.center_pos, 127, crop_size, self.channel_average)

        self.model.init(z_crop)

    def track(self, img):
        w_z = self.size[0] + self.context_amount * np.sum(self.size)
        h_z = self.size[1] + self.context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = 127 / s_z
        s_x = s_z * (255 / 127)
        x_crop = get_subwindow_tracking(img, self.center_pos, 255, round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        # print(pred_bbox)

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
        penalty = np.exp(-(r_c * s_c - 1) * 0.138)

        # score
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 -  0.455) + \
                 self.window * 0.455

        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * 0.348

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

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }


    def on_mouse(self, event, cx, cy, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (cx, cy, w, h)
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
