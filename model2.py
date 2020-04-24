import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 4 spectrograms as input
        # [:, 4, 128, 44]
        self.conv1_1 = nn.Conv2d(4, 8, kernel_size=(5, 3))
        self.conv1_2 = nn.Conv2d(8, 12, kernel_size=(5, 3))
        self.f_pool1 = nn.MaxPool2d((2, 2))
        # [:, 12, 59, 20]
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2_1 = nn.Conv2d(12, 16, kernel_size=(4, 3))
        self.conv2_2 = nn.Conv2d(16, 24, kernel_size=(4, 3))
        self.f_pool2 = nn.MaxPool2d((2, 1))
        # [:, 24, 25, 16]
        self.bn2 = nn.BatchNorm2d(24)

        self.conv3_1 = nn.Conv2d(24, 36, kernel_size=(3, 3))
        self.conv3_2 = nn.Conv2d(36, 48, kernel_size=(3, 3))
        self.f_pool3 = nn.MaxPool2d((2, 1))
        # [:, 48, 9, 12]
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4_1 = nn.Conv2d(48, 64, kernel_size=(3, 3))
        self.conv4_2 = nn.Conv2d(64, 96, kernel_size=(3, 3))
        # self.f_pool4 = nn.MaxPool2d((2, 1))
        # [:, 96, 2, 8]
        self.bn4 = nn.BatchNorm2d(96)

        self.conv5_1 = nn.Conv2d(96, 128, kernel_size=(3, 3))
        # self.conv5_2 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        # self.f_pool4 = nn.MaxPool2d((2, 1))
        # [:, 96, 2, 8]
        self.bn5 = nn.BatchNorm2d(128)

        # self.conv5 = nn.Conv2d(96, 96, kernel_size=(2, 1))
        # [:, 96, 1, 8]

        # each head produces a gain mask for a dedicated track
        self.conv_head1_1 = nn.Conv2d(128, 16, kernel_size=(3, 3))
        self.conv_head1_2 = nn.Conv2d(16, 1, kernel_size=(3, 1))

        self.conv_head2_1 = nn.Conv2d(128, 16, kernel_size=(3, 3))
        self.conv_head2_2 = nn.Conv2d(16, 1, kernel_size=(3, 1))

        self.conv_head3_1 = nn.Conv2d(128, 16, kernel_size=(3, 3))
        self.conv_head3_2 = nn.Conv2d(16, 1, kernel_size=(3, 1))

        self.conv_head4_1 = nn.Conv2d(128, 16, kernel_size=(3, 3))
        self.conv_head4_2 = nn.Conv2d(16, 1, kernel_size=(3, 1))

    # @staticmethod
    # def _normalize_tensor(x):
    #     x -= x.min(0, keepdim=True)[0]
    #     x = x / (x.max(0, keepdim=True)[0] + 0.00001)
    #     x = 2 * x - 1
    #     return x

    def forward(self, x):
        """

        :param x:
        :return: sum of masked track spectrograms +
        mask for each track to apply them to audio during test time
        """
        # [:, 4, 128, 44]
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.f_pool1(res)
        res = self.bn1(res)
        # print(res.shape)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.f_pool2(res)
        res = self.bn2(res)
        # print(res.shape)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = self.f_pool3(res)
        res = self.bn3(res)
        # print(res.shape)

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        # res = self.f_pool4(res)
        res = self.bn4(res)
        # print(res.shape)

        res = F.relu(self.conv5_1(res))
        # res = F.relu(self.conv5_2(res))
        res = self.bn5(res)
        # print(res.shape)

        # res = F.relu(self.conv5(res))
        # print(res.shape)

        m1 = F.relu(self.conv_head1_1(res))
        # print('m1: ', m1.shape)
        m1 = self.conv_head1_2(m1)
        # m1 = F.interpolate(self.conv_head1_2(m1), size=44)
        # print('m1: ', m1.shape)
        m1 = F.interpolate(m1.view((-1, 1, 4)), size=x.shape[-1])
        # print('m1: ', m1.shape)

        m2 = F.relu(self.conv_head2_1(res))
        m2 = self.conv_head2_2(m2)
        m2 = F.interpolate(m2.view((-1, 1, 4)), size=x.shape[-1])

        m3 = F.relu(self.conv_head3_1(res))
        m3 = self.conv_head3_2(m3)
        m3 = F.interpolate(m3.view((-1, 1, 4)), size=x.shape[-1])

        m4 = F.relu(self.conv_head4_1(res))
        m4 = self.conv_head4_2(m4)
        m4 = F.interpolate(m4.view((-1, 1, 4)), size=x.shape[-1])

        # x2 = F.interpolate(self.conv_head2(res), size=44)
        # x3 = F.interpolate(self.conv_head3(res), size=44)
        # x4 = F.interpolate(self.conv_head4(res), size=44)

        # print(x1.view((-1, 44)).shape)

        masked = torch.zeros_like(x[:, 0])
        masked += m1 * x[:, 0]
        masked += m2 * x[:, 1]
        masked += m3 * x[:, 2]
        masked += m4 * x[:, 3]
        # masked += self._normalize_tensor(x1) * x[:, 0]
        # masked += self._normalize_tensor(x2) * x[:, 1]
        # masked += self._normalize_tensor(x3) * x[:, 2]
        # masked += self._normalize_tensor(x4) * x[:, 3]

        # print(masked.shape)

        return masked, tuple(elem.view((-1, 44)) for elem in (m1, m2, m3, m4))
