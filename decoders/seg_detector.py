from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


###
def dw_conv(in_planes, stride=1):
    """
    depth wise卷积核
    pytorch实现depth wise卷积核非常简单，只需要将输入通道和输出通道赋值相同以及groups赋值为输入通道即可
    """
    return nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)


def pw_conv(in_planes, out_planes):
    """
    point wise卷积核
    point wise卷积核就是普通的1x1的卷积核
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


class DPBlock(nn.Module):

    def __init__(self, planes, stride=1):
        super(DPBlock, self).__init__()
        self.dw_conv = dw_conv(planes, stride)
        self.pw_conv = pw_conv(planes, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


###

class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        ###
        self.up_dp1 = self.make_dp_module(DPBlock, 256)
        self.up_dp2 = self.make_dp_module(DPBlock, 256)
        self.up_dp3 = self.make_dp_module(DPBlock, 256)

        self.down_dp1 = self.make_dp_module(DPBlock, 256, stride=2)
        self.down_dp2 = self.make_dp_module(DPBlock, 256, stride=2)
        self.down_dp3 = self.make_dp_module(DPBlock, 256, stride=2)
        ###
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        self.up_dp1.apply(self.weights_init)
        self.up_dp2.apply(self.weights_init)
        self.up_dp3.apply(self.weights_init)
        self.down_dp1.apply(self.weights_init)
        self.down_dp2.apply(self.weights_init)
        self.down_dp3.apply(self.weights_init)



    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    ###

    def make_dp_module(self, block, planes, stride=1):
        return block(planes, stride)

    def make_fpem(self, feature_map_list):
        up_en_p5 = feature_map_list[0]

        up_en_p4 = self._upsample_add(up_en_p5, feature_map_list[1])
        up_en_p4 = self.up_dp1(up_en_p4)

        up_en_p3 = self._upsample_add(up_en_p4, feature_map_list[2])
        up_en_p3 = self.up_dp2(up_en_p3)

        up_en_p2 = self._upsample_add(up_en_p3, feature_map_list[3])
        up_en_p2 = self.up_dp3(up_en_p2)

        down_en_p2 = up_en_p2

        down_en_p3 = self._upsample_add(up_en_p3, down_en_p2)
        down_en_p3 = self.down_dp1(down_en_p3)

        down_en_p4 = self._upsample_add(up_en_p4, down_en_p3)
        down_en_p4 = self.down_dp2(down_en_p4)

        down_en_p5 = self._upsample_add(up_en_p5, down_en_p4)
        down_en_p5 = self.down_dp3(down_en_p5)

        return down_en_p5, down_en_p4, down_en_p3, down_en_p2

    ###

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear') + y

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        for i in range(2):
            former_p5 = in5
            former_p4 = in4
            former_p3 = in3
            former_p2 = in2
            in5, in4, in3, in2 = self.make_fpem([in5, in4, in3, in2])
            if i == 0:
                out_p5 = in5
                out_p4 = in4
                out_p3 = in3
                out_p2 = in2
            else:
                out_p5 = former_p5 + in5
                out_p4 = former_p4 + in4
                out_p3 = former_p3 + in3
                out_p2 = former_p2 + in2

        out_p5 = self.out5(out_p5)
        out_p4 = self.out4(out_p4)
        out_p3 = self.out3(out_p3)
        out_p2 = self.out2(out_p2)

        fuse = torch.cat((out_p2, out_p3, out_p4, out_p5), 1)
        '''
        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        

        fuse = torch.cat((p5, p4, p3, p2), 1)
        '''
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(
                        binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
