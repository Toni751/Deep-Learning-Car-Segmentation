import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            # We use bias=False because it is somehow cancelled out by the batchnorm
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # output shape excluding channels (same for both height and width) is:
        # out = (in - 1) * stride - 2 * padding + (kernel_size - 1) + 1
        # here, with padding = 0, we get:
        # out = (stride * in) - (2 * stride) + kernel
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, out_channels)

    def forward(self, x1, x2, x3=None, x4=None, x5=None):
        x1 = self.up(x1)
        tensors = [x2]
        if x3 is not None:
            tensors.append(x3)
        if x4 is not None:
            tensors.append(x4)
        if x5 is not None:
            tensors.append(x5)

        tensors.append(x1)
        x = torch.cat(tensors, dim=1)  # all tensors need to have the same number of channels
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # the CrossEntropyLoss automatically wraps this in a LogSoftmax, that's why we don't do it here
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = (ConvBlock(3, 64, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, 9))

    def forward(self, x):
        x1 = self.inc(x)  # x1 HxW: 256x256
        x2 = self.down1(x1)  # x2 HxW: 128x128
        x3 = self.down2(x2)  # x3 HxW: 64x64
        x4 = self.down3(x3)  # x4 HxW: 32x32
        x5 = self.down4(x4)  # x5 HxW: 16x16
        x = self.up1(x5, x4)  # up(x5) gives 32x32, concat with x4, HxW remains 32x32 and the channels are added
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0_0 = ConvBlock(3, 32, 32)
        self.down1_0 = Down(32, 64)
        self.down2_0 = Down(64, 128)
        self.down3_0 = Down(128, 256)
        self.down4_0 = Down(256, 512)

        self.up0_1 = Up(32 + 64, 32)
        self.up1_1 = Up(64 + 128, 64)
        self.up2_1 = Up(128 + 256, 128)
        self.up3_1 = Up(256 + 512, 256)

        self.up0_2 = Up(32 * 2 + 64, 32)
        self.up1_2 = Up(64 * 2 + 128, 64)
        self.up2_2 = Up(128 * 2 + 256, 128)

        self.up0_3 = Up(32 * 3 + 64, 32)
        self.up1_3 = Up(64 * 3 + 128, 64)

        self.up0_4 = Up(32 * 4 + 64, 32)

        self.outc = OutConv(32, 9)

    def forward(self, input):
        x0_0 = self.conv0_0(input)  # 256x256
        x1_0 = self.down1_0(x0_0)  # 128x128
        x0_1 = self.up0_1(x1_0, x0_0)  # 256x256

        x2_0 = self.down2_0(x1_0)  # 64x64
        x1_1 = self.up1_1(x2_0, x1_0)  # 128x128
        x0_2 = self.up0_2(x1_1, x0_0, x0_1)  # 256x256

        x3_0 = self.down3_0(x2_0)  # 32x32
        x2_1 = self.up2_1(x3_0, x2_0)  # 64x64
        x1_2 = self.up1_2(x2_1, x1_0, x1_1)  # 128x128
        x0_3 = self.up0_3(x1_2, x0_0, x0_1, x0_2)  # 256x256

        x4_0 = self.down4_0(x3_0)  # 16x16
        x3_1 = self.up3_1(x4_0, x3_0)  # 32x32
        x2_2 = self.up2_2(x3_1, x2_0, x2_1)  # 64x64
        x1_3 = self.up1_3(x2_2, x1_0, x1_1, x1_2)  # 128x128
        x0_4 = self.up0_4(x1_3, x0_0, x0_1, x0_2, x0_3)  # 256x256

        logits = self.outc(x0_4)
        return logits
