import torch.nn as nn
import torch.nn.functional as F


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po



class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class UPChannelBAN(BAN):
    def __init__(self, feature_in: int = 96):
        super(UPChannelBAN, self).__init__()
        loc_output = 4
        self.template_loc_conv = nn.Conv2d(feature_in, feature_in * loc_output, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, template_features, search_features):
        loc_kernel = self.template_loc_conv(template_features)
        loc_feature = self.search_loc_conv(search_features)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return loc
