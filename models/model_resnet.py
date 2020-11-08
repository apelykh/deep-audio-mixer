import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 96, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 128, num_blocks[4], stride=2)
        self.layer6 = self._make_layer(block, 256, num_blocks[5], stride=2)

        flattened_dim = 231

        self.conv_head1 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head1 = nn.Linear(flattened_dim, 1)

        self.conv_head2 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head2 = nn.Linear(flattened_dim, 1)

        self.conv_head3 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head3 = nn.Linear(flattened_dim, 1)

        self.conv_head4 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.fc_head4 = nn.Linear(flattened_dim, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        m1 = F.relu(self.conv_head1(out))
        m1 = self.fc_head1(m1.view((x.size()[0], -1)))

        m2 = F.relu(self.conv_head2(out))
        m2 = self.fc_head2(m2.view((x.size()[0], -1)))

        m3 = F.relu(self.conv_head3(out))
        m3 = self.fc_head3(m3.view((x.size()[0], -1)))

        m4 = F.relu(self.conv_head4(out))
        m4 = self.fc_head4(m4.view((x.size()[0], -1)))

        masked = torch.zeros_like(x[:, 0])
        masked += m1.unsqueeze(2) * x[:, 0]
        masked += m2.unsqueeze(2) * x[:, 1]
        masked += m3.unsqueeze(2) * x[:, 2]
        masked += m4.unsqueeze(2) * x[:, 3]

        return masked, (m1, m2, m3, m4)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
#
# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
#
#
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])
#
#
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def run_dummy():
    net = ResNet18().to('cuda')

    num_trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('{} trainable parameters'.format(num_trainable_param))

    masked, masks = net(torch.randn(16, 4, 1025, 216).to('cuda'))
    print(masked.size())


if __name__ == '__main__':
    run_dummy()
