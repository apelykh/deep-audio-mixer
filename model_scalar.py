import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModelScalar(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 4 spectrograms as input
        self.conv1_1 = nn.Conv2d(4, 8, kernel_size=(5, 3))
        self.conv1_2 = nn.Conv2d(8, 12, kernel_size=(5, 3))
        self.f_pool1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2_1 = nn.Conv2d(12, 16, kernel_size=(4, 3))
        self.conv2_2 = nn.Conv2d(16, 24, kernel_size=(4, 3))
        self.f_pool2 = nn.MaxPool2d((2, 1))
        self.bn2 = nn.BatchNorm2d(24)

        self.conv3_1 = nn.Conv2d(24, 36, kernel_size=(3, 3))
        self.conv3_2 = nn.Conv2d(36, 48, kernel_size=(3, 3))
        self.f_pool3 = nn.MaxPool2d((2, 1))
        self.bn3 = nn.BatchNorm2d(48)

        # each head produces a gain coefficient for a dedicated track
        self.conv_head1 = nn.Conv2d(48, 1, kernel_size=(1, 1))
        self.fc_head1 = nn.Linear(1476, 1)

        self.conv_head2 = nn.Conv2d(48, 1, kernel_size=(1, 1))
        self.fc_head2 = nn.Linear(1476, 1)

        self.conv_head3 = nn.Conv2d(48, 1, kernel_size=(1, 1))
        self.fc_head3 = nn.Linear(1476, 1)

        self.conv_head4 = nn.Conv2d(48, 1, kernel_size=(1, 1))
        self.fc_head4 = nn.Linear(1476, 1)

    def forward(self, x):
        """

        :param x:
        :return: sum of masked track spectrograms +
        mask for each track to apply them to audio during test time
        """
        # [:, 4, 1025, 44]
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.f_pool1(res)
        res = self.bn1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.f_pool2(res)
        res = self.bn2(res)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = self.f_pool3(res)
        res = self.bn3(res)

        m1 = F.relu(self.conv_head1(res))
        m1 = self.fc_head1(m1.view((x.size()[0], -1)))

        m2 = F.relu(self.conv_head2(res))
        m2 = self.fc_head2(m2.view((x.size()[0], -1)))

        m3 = F.relu(self.conv_head3(res))
        m3 = self.fc_head3(m3.view((x.size()[0], -1)))

        m4 = F.relu(self.conv_head4(res))
        m4 = self.fc_head4(m4.view((x.size()[0], -1)))

        ones = torch.ones_like(m1)

        masked = torch.zeros_like(x[:, 0])
        masked += (ones + m1).unsqueeze(2) * x[:, 0]
        masked += (ones + m2).unsqueeze(2) * x[:, 1]
        masked += (ones + m3).unsqueeze(2) * x[:, 2]
        masked += (ones + m4).unsqueeze(2) * x[:, 3]

        return masked, (m1, m2, m3, m4)
