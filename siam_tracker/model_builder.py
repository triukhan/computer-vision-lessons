from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from siam_tracker.backbone import mobilenetv3_small_v3
from siam_tracker.head import UPChannelBAN


# from nanotrack.core.config import cfg


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # self.cfg = cfg

        self.backbone = mobilenetv3_small_v3()
        self.ban_head = UPChannelBAN()
        # self.neck = None
        # if cfg.ADJUST.ADJUST:
        #     self.neck = get_neck(cfg.ADJUST.TYPE,
        #                          **cfg.ADJUST.KWARGS)
        self.template_features = None

    def init(self, z):
        template_features = self.backbone(z)
        self.template_features = template_features

    def track(self, x):
        search_features = self.backbone(x)
        loc = self.ban_head(self.template_features, search_features)

        return {'loc': loc}


    def forward(self, data):
        """
            only used in training
        """
        # train mode
        if len(data) >= 4:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_loc = data['label_loc'].cuda()

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)

            # if self.neck is not None:
            #     cls, reg = self.neck(xf, zf)

            loc = self.ban_head(zf, xf)

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc)
            outputs = {}

            outputs['total_loss'] = self.cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['loc_loss'] = loc_loss

            return outputs
        else:
            xf = self.backbone(data)
            loc = self.ban_head(self.zf, xf)

            return {'loc': loc}
