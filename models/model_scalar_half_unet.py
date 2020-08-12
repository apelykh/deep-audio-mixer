from models.unet_parts import *


class MixingModelScalar(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = DoubleConvDilated(4, 32, strided=True)
        self.down1 = DoubleConvDilated(32, 64, strided=True)
        self.down2 = DoubleConvDilated(64, 128, strided=True)
        self.down3 = DoubleConvDilated(128, 256, strided=True)
        # self.down4 = DoubleConvDilated(256, 512)

        # each head produces a gain coefficient for a dedicated track
        self.conv_head1 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head1 = nn.Linear(78, 1)

        self.conv_head2 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head2 = nn.Linear(78, 1)

        self.conv_head3 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head3 = nn.Linear(78, 1)

        self.conv_head4 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head4 = nn.Linear(78, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        res = self.inc(x)
        res = self.down1(res)
        res = self.down2(res)
        res = self.down3(res)
        # res = self.down4(res)

        m1 = F.relu(self.conv_head1(res))
        m1 = self.fc_head1(m1.view((x.size()[0], -1)))
        print(m1.shape)

        m2 = F.relu(self.conv_head2(res))
        m2 = self.fc_head2(m2.view((x.size()[0], -1)))

        m3 = F.relu(self.conv_head3(res))
        m3 = self.fc_head3(m3.view((x.size()[0], -1)))

        m4 = F.relu(self.conv_head4(res))
        m4 = self.fc_head4(m4.view((x.size()[0], -1)))

        masked = torch.zeros_like(x[:, 0])
        masked += m1.unsqueeze(2) * x[:, 0]
        masked += m2.unsqueeze(2) * x[:, 1]
        masked += m3.unsqueeze(2) * x[:, 2]
        masked += m4.unsqueeze(2) * x[:, 3]

        return masked, (m1, m2, m3, m4)
