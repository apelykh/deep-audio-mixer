import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 4 spectrograms as input
        # [:, 4, 128, 44]
        self.conv1_1 = nn.Conv2d(4, 8, kernel_size=(6, 3))
        self.conv1_2 = nn.Conv2d(8, 12, kernel_size=(6, 3))
        self.f_pool1 = nn.MaxPool2d((2, 2))
        # [:, 12, 59, 20]

        self.conv2_1 = nn.Conv2d(12, 16, kernel_size=(5, 3))
        self.conv2_2 = nn.Conv2d(16, 24, kernel_size=(5, 3))
        self.f_pool2 = nn.MaxPool2d((2, 1))
        # [:, 24, 25, 16]

        self.conv3_1 = nn.Conv2d(24, 36, kernel_size=(4, 3))
        self.conv3_2 = nn.Conv2d(36, 48, kernel_size=(4, 3))
        self.f_pool3 = nn.MaxPool2d((2, 1))
        # [:, 48, 9, 12]

        self.conv4_1 = nn.Conv2d(48, 48, kernel_size=(5, 1))
        self.conv4_2 = nn.Conv2d(48, 48, kernel_size=(5, 1))
        # [:, 48, 1, 12]

        # each head produces a gain mask for a dedicated track
        self.conv_head1 = nn.Conv1d(48, 1, 3)
        self.conv_head2 = nn.Conv1d(48, 1, 3)
        self.conv_head3 = nn.Conv1d(48, 1, 3)
        self.conv_head4 = nn.Conv1d(48, 1, 3)

    @staticmethod
    def _normalize_tensor(x):
        x -= x.min(0, keepdim=True)[0]
        x = x / (x.max(0, keepdim=True)[0] + 0.00001)
        x = 2 * x - 1
        return x

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
        # print(res.shape)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.f_pool2(res)
        # print(res.shape)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = self.f_pool3(res)
        # print(res.shape)

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        # print(res.shape)

        res = res.view((-1, 48, 12))

        x1 = F.interpolate(self.conv_head1(res), size=x.shape[-1])
        x2 = F.interpolate(self.conv_head2(res), size=x.shape[-1])
        x3 = F.interpolate(self.conv_head3(res), size=x.shape[-1])
        x4 = F.interpolate(self.conv_head4(res), size=x.shape[-1])

        masked = torch.zeros_like(x[:, 0])
        masked += x1 * x[:, 0]
        masked += x2 * x[:, 1]
        masked += x3 * x[:, 2]
        masked += x4 * x[:, 3]
        # masked += self._normalize_tensor(x1) * x[:, 0]
        # masked += self._normalize_tensor(x2) * x[:, 1]
        # masked += self._normalize_tensor(x3) * x[:, 2]
        # masked += self._normalize_tensor(x4) * x[:, 3]

        # print(masked.shape)

        return masked, tuple(elem.view((-1, 44)) for elem in (x1, x2, x3, x4))
