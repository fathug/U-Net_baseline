# -*- coding:utf-8 -*-
# @Time : 2022/5/27 15:33
# @Author : 
# @Note :
import torch
from torch import nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, padding_mode='reflect',
                      bias='False'),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):  # 考虑到最大池化会丢失很多特征，所以采用卷积
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias='False'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)
        )

    def forward(self, x, feature_map):  # 先变宽（把图变大），再变短（降通道）
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)

        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)

        self.out = nn.Conv2d(64, 3, (3, 3), 1, 1)

    def forward(self, x):
        out1 = self.c1(x)
        out2 = self.c2(self.d1(out1))
        out3 = self.c3(self.d2(out2))
        out4 = self.c4(self.d3(out3))
        out5 = self.c5(self.d4(out4))

        u1 = self.c6(self.u1(out5, out4))
        u2 = self.c7(self.u2(u1, out3))
        u3 = self.c8(self.u3(u2, out2))
        u4 = self.c9(self.u4(u3, out1))

        final = nn.Sigmoid()(self.out(u4))
        return final


# unet = UNet()
# if __name__ == '__main__':
#     x = torch.randn(2, 3, 256, 256)  # 喂入随机数据，测试输出形状
#     print(unet(x).shape)

# unet = UNet()
# if __name__ == '__main__':
#     print(unet)