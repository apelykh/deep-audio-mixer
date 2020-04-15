import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 5 spectrograms as input
        # [:, 5, 128, 216]
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.f_pool1 = nn.MaxPool2d((2, 1))
        # [:, 8, 64, 216]
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.f_pool2 = nn.MaxPool2d((2, 1))
        # [:, 16, 32, 216]
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.f_pool3 = nn.MaxPool2d((2, 1))
        # [:, 32, 16, 216]
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.f_pool4 = nn.MaxPool2d((2, 1))
        # [:, 64, 8, 216]
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.f_pool5 = nn.MaxPool2d((2, 1))
        # [:, 128, 4, 216]
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(4, 1))
        # [:, 128, 1, 216]

        # each head produces a gain mask for a dedicated track
        self.conv_head1 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head2 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head3 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head4 = nn.Conv1d(128, 1, 3, padding=1)

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
        # print(x.shape)
        res = F.relu(self.conv1(x))
        res = self.f_pool1(res)
        # print(res.shape)
        res = F.relu(self.conv2(res))
        res = self.f_pool2(res)
        # print(res.shape)
        res = F.relu(self.conv3(res))
        res = self.f_pool3(res)
        # print(res.shape)
        res = F.relu(self.conv4(res))
        res = self.f_pool4(res)
        # print(res.shape)
        res = F.relu(self.conv5(res))
        res = self.f_pool5(res)
        # print(res.shape)
        res = F.relu(self.conv6(res))
        # print(res.shape)

        res = res.view((-1, 128, 216))
        # print(res.shape)

        x1 = self.conv_head1(res)
        x2 = self.conv_head2(res)
        x3 = self.conv_head3(res)
        x4 = self.conv_head4(res)

        masked = torch.zeros_like(x[:, 0])
        masked += x1 * x[:, 0]
        masked += x2 * x[:, 1]
        masked += x3 * x[:, 2]
        masked += x4 * x[:, 3]
        # masked += self._normalize_tensor(x1) * x[:, 0]
        # masked += self._normalize_tensor(x2) * x[:, 1]
        # masked += self._normalize_tensor(x3) * x[:, 2]
        # masked += self._normalize_tensor(x4) * x[:, 3]

        return masked, tuple(elem.view((-1, 216)) for elem in (x1, x2, x3, x4))
