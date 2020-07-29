from models.unet_parts import *


class ModelUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        self.inc = DoubleConvDilated(n_channels, 64, strided=True)
        self.down1 = DoubleConvDilated(64, 128, strided=True)
        self.down2 = DoubleConvDilated(128, 256, strided=True)
        self.down3 = DoubleConvDilated(256, 512, strided=True)
        self.down4 = DoubleConvDilated(512, 1024)

        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        logits = self.outc(out)

        logits = logits[:, :, :, :-1]

        masked = torch.zeros_like(x[:, 0])
        masked += logits[:, 0] * x[:, 0]
        masked += logits[:, 1] * x[:, 1]
        masked += logits[:, 2] * x[:, 2]
        masked += logits[:, 3] * x[:, 3]

        return masked
