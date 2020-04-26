import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModelVector(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 4 spectrograms as input
        self.conv1_1 = nn.Conv2d(4, 8, kernel_size=(5, 5))
        self.conv1_2 = nn.Conv2d(8, 12, kernel_size=(5, 5))
        self.f_pool1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2_1 = nn.Conv2d(12, 16, kernel_size=(4, 4))
        self.conv2_2 = nn.Conv2d(16, 24, kernel_size=(4, 4))
        self.f_pool2 = nn.MaxPool2d((2, 1))
        self.bn2 = nn.BatchNorm2d(24)

        self.conv3_1 = nn.Conv2d(24, 36, kernel_size=(3, 3))
        self.conv3_2 = nn.Conv2d(36, 48, kernel_size=(3, 3))
        self.f_pool3 = nn.MaxPool2d((2, 1))
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4_1 = nn.Conv2d(48, 64, kernel_size=(5, 1))
        self.conv4_2 = nn.Conv2d(64, 72, kernel_size=(5, 1))
        self.f_pool4 = nn.MaxPool2d((2, 1))
        self.bn4 = nn.BatchNorm2d(72)

        self.conv5 = nn.Conv2d(72, 72, kernel_size=(5, 1))
        self.f_pool5 = nn.MaxPool2d((2, 1))

        self.conv6 = nn.Conv2d(72, 72, kernel_size=(5, 1))
        self.f_pool6 = nn.MaxPool2d((2, 1))

        # each head produces a gain mask for a dedicated track
        self.conv_head1 = nn.Conv1d(72, 1, kernel_size=(11, 1))
        self.conv_head2 = nn.Conv1d(72, 1, kernel_size=(11, 1))
        self.conv_head3 = nn.Conv1d(72, 1, kernel_size=(11, 1))
        self.conv_head4 = nn.Conv1d(72, 1, kernel_size=(11, 1))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        :param x: input features;
        :return:
        - weighted sum (gain masks applied) of input spectrograms;
        - gain masks for each stem;
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

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        res = self.f_pool4(res)
        res = self.bn4(res)

        res = F.relu(self.conv5(res))
        res = self.f_pool5(res)

        res = F.relu(self.conv6(res))
        res = self.f_pool6(res)

        h1 = self.conv_head1(res)
        h2 = self.conv_head2(res)
        h3 = self.conv_head3(res)
        h4 = self.conv_head4(res)

        # interpolate masks to a length of the initial spectrogram
        m1 = F.interpolate(h1.view(-1, 1, 8), size=x.shape[-1], mode='linear')
        m2 = F.interpolate(h2.view(-1, 1, 8), size=x.shape[-1], mode='linear')
        m3 = F.interpolate(h3.view(-1, 1, 8), size=x.shape[-1], mode='linear')
        m4 = F.interpolate(h4.view(-1, 1, 8), size=x.shape[-1], mode='linear')

        masked = torch.zeros_like(x[:, 0])
        masked += m1 * x[:, 0]
        masked += m2 * x[:, 1]
        masked += m3 * x[:, 2]
        masked += m4 * x[:, 3]

        return masked, tuple(elem.view((-1, x.shape[-1])) for elem in (m1, m2, m3, m4))
