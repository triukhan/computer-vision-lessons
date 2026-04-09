from pathlib import Path

import cv2
import torch

from siam_tracker.model_builder import ModelBuilder
from siam_tracker.nano_tracker import NanoTracker

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nanotrackv3.pth'

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print('[Warning] missing keys: {}'.format(missing_keys))
        print('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        print('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        print('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only=False)

    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        print('[Warning]: using pretrain as features. Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def track_object(video_path: Path, stop=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = load_pretrain(ModelBuilder(), MODEL_PATH).eval()
    model.eval()

    tracker = NanoTracker(model)
    cv2.namedWindow('tracking with siam', cv2.WINDOW_NORMAL)
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
            x, y, w, h = tracker.bbox
            cv2.rectangle(
                frame,
                (int(tracker.center_pos[0] - 60 / 2), int(tracker.center_pos[1] - 60 / 2)),
                (int(tracker.center_pos[0] - 60 / 2) + 60, int(tracker.center_pos[1] - 60 / 2) + 60),
                (0, 255, 0),
                2
            )
            # cv2.rectangle(
            #     frame,
            #     (int(x), int(y)),
            #     (int(x + w), int(y + h)),
            #     (0, 0, 255),
            #     2
            # )

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'tracker_project' / 'helicopter.mp4'
track_object(vid, stop=True)
