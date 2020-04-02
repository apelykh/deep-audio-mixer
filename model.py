import torch.nn as nn
import torch.nn.functional as F


class MixingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # stack of 5 spectrograms as input
        # [1, 5, 128, 216]
        self.conv1 = nn.Conv2d(5, 8, 3, padding=1)
        self.f_pool1 = nn.MaxPool2d((2, 1))
        # [1, 8, 64, 216]
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.f_pool2 = nn.MaxPool2d((2, 1))
        # [1, 16, 32, 216]
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.f_pool3 = nn.MaxPool2d((2, 1))
        # [1, 32, 16, 216]
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.f_pool4 = nn.MaxPool2d((2, 1))
        # [1, 64, 8, 216]
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.f_pool5 = nn.MaxPool2d((2, 1))
        # [1, 128, 4, 216]
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2)
        # [1, 128, 1, 107]

        # each head produces a gain mask for a dedicated track
        self.conv_head1 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head2 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head3 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head4 = nn.Conv1d(128, 1, 3, padding=1)
        self.conv_head5 = nn.Conv1d(128, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.f_pool1(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.f_pool2(x)
        print(x.shape)
        x = F.relu(self.conv3(x))
        x = self.f_pool3(x)
        print(x.shape)
        x = F.relu(self.conv4(x))
        x = self.f_pool4(x)
        print(x.shape)
        x = F.relu(self.conv5(x))
        x = self.f_pool5(x)
        print(x.shape)
        x = F.relu(self.conv6(x))
        print(x.shape)

        x = x.view((-1, 128, 107))
        print(x.shape)

        x1 = self.conv_head1(x)
        x2 = self.conv_head2(x)
        x3 = self.conv_head3(x)
        x4 = self.conv_head4(x)
        x5 = self.conv_head5(x)

        return tuple(elem.view((-1, 107)) for elem in (x1, x2, x3, x4, x5))
