from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from re import template

import torch.nn as nn

# from siam_tracker.backbone import mobilenetv3_small_v3
from siam_tracker.head import UPChannelBAN

from torchvision.models import mobilenet_v3_small

# from nanotrack.core.config import cfg


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # self.cfg = cfg

        self.backbone = mobilenet_v3_small(weights="DEFAULT").features[:7]
        self.ban_head = UPChannelBAN()
        # self.neck = None
        # if cfg.ADJUST.ADJUST:
        #     self.neck = get_neck(cfg.ADJUST.TYPE,
        #                          **cfg.ADJUST.KWARGS)
        self.template_features = None
        self.prev_feat = None

        self.neck = nn.Conv2d(40, 96, 1)  # підлаштувати канали

    def init(self, z):
        template_features = self.neck(self.backbone(z))
        self.template_features = template_features

    def track(self, x):
        # search_features = self.backbone(x)

        search_features = self.neck(self.backbone(x))

        if self.prev_feat is not None:
            print("feat diff:", (search_features - self.prev_feat).abs().mean())

        loc = self.ban_head(self.template_features, search_features)
        self.prev_feat = search_features

        return {'loc': loc}


    def forward(self, data):
        """
            only used in training
        """
        # train mode
        # if len(data) >= 4:
        #     template = data['template'].cuda()
        #     search = data['search'].cuda()
        #     label_loc = data['label_loc'].cuda()
        #
        #     # get feature
        #     zf = self.backbone(template)
        #     xf = self.backbone(search)
        #
        #     # if self.neck is not None:
        #     #     cls, reg = self.neck(xf, zf)
        #
        #     loc = self.ban_head(zf, xf)
        #
        #     # loc loss with iou loss
        #     loc_loss = select_iou_loss(loc, label_loc)
        #     outputs = {}
        #
        #     outputs['total_loss'] = self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        #     outputs['loc_loss'] = loc_loss
        #
        #     return outputs
        # else:
        #     xf = self.backbone(data)
        #     loc = self.ban_head(self.zf, xf)
        #
        #     return {'loc': loc}
        pass
