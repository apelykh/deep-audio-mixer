import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingModelTDD(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(4, 16, kernel_size=(5, 5))
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(5, 5))
        self.f_pool1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(16, 26, kernel_size=(5, 3))
        self.conv2_2 = nn.Conv2d(26, 24, kernel_size=(5, 3))
        self.f_pool2 = nn.MaxPool2d((2, 1))
        self.bn2 = nn.BatchNorm2d(24)

        self.conv3_1 = nn.Conv2d(24, 48, kernel_size=(3, 3))
        self.conv3_2 = nn.Conv2d(48, 48, kernel_size=(3, 3))
        self.f_pool3 = nn.MaxPool2d((2, 1))
        self.bn3 = nn.BatchNorm2d(48)

        # https://stackoverflow.com/questions/61372645/how-to-implement-time-distributed-dense-tdd-layer-in-pytorch/61372646
        # num_of_input_channels - height of the freq axis
        self.tdd1 = nn.Conv2d(48, 1, (123, 1))
        self.tdd2 = nn.Conv2d(48, 1, (123, 1))
        self.tdd3 = nn.Conv2d(48, 1, (123, 1))
        self.tdd4 = nn.Conv2d(48, 1, (123, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, 4, 1025, 87)
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

        h1 = self.tdd1(res)
        h2 = self.tdd2(res)
        h3 = self.tdd3(res)
        h4 = self.tdd4(res)
        # print(h1.shape)

        # interpolate masks to a length of the initial spectrogram
        m1 = F.interpolate(h1.view(-1, 1, 31), size=x.shape[-1], mode='linear')
        m2 = F.interpolate(h2.view(-1, 1, 31), size=x.shape[-1], mode='linear')
        m3 = F.interpolate(h3.view(-1, 1, 31), size=x.shape[-1], mode='linear')
        m4 = F.interpolate(h4.view(-1, 1, 31), size=x.shape[-1], mode='linear')
        # print(m1.shape)

        masked = torch.zeros_like(x[:, 0])
        masked += m1 * x[:, 0]
        masked += m2 * x[:, 1]
        masked += m3 * x[:, 2]
        masked += m4 * x[:, 3]
        # print(masked.shape)

        return masked
