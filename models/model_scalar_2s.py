import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import MultitrackAudioDataset
from data.medleydb_split import weathervane_music


class ConvBlock2d(nn.Module):
    """
    Implementation of a 2D convolutional block of the following structure:
    Conv2D -> BatchNorm -> ReLU -> Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, dropout_p: float = -1.0):
        super().__init__()
        # 'same' padding strategy
        # out_rows = (in_channels + stride - 1) // stride
        # padding_size = max(0, (out_rows - 1) * stride + (kernel_size - 1) * dilation + 1 - in_channels)
        #
        # self.add_padding = nn.ReflectionPad2d(padding_size // 2) if padding_size > 0 else None

        self.add_padding = None

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              dilation=dilation)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                         momentum=0.90,
                                         eps=0.001)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p) if dropout_p != -1 else None

    def forward(self, x):
        if self.add_padding:
            x = self.add_padding(x)

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        if self.training and self.dropout:
            out = self.dropout(out)

        return out


def dB_to_amplitude(x: torch.Tensor):
    """
    db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db)
    """
    return torch.pow(10.0, 0.5 * x)


def amplitude_to_dB(x: torch.Tensor):
    """
    amplitude_to_dB(S) = 20 * log10(S)
    """
    return 20 * torch.log10(x)


class MixingModelScalar2s(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_b1 = ConvBlock2d(4, 16, 3, dropout_p=0.2, stride=2, dilation=2)
        self.conv_b2 = ConvBlock2d(16, 32, 5, dropout_p=0.2)
        self.conv_b3 = ConvBlock2d(32, 48, 5, dropout_p=0.2)
        self.conv_b4 = ConvBlock2d(48, 64, 7, dropout_p=0.2)
        self.conv_b5 = ConvBlock2d(64, 128, 9, dropout_p=0.3)
#         self.conv_b6 = ConvBlock2d(96, 128, 9, dropout_p=0.3)
        # self.conv6 = nn.Conv2d(in_channels=316, out_channels=512, kernel_size=1)
        # self.conv7 = nn.Conv2d(in_channels=196, out_channels=4, kernel_size=1)

        flattened_dim = 30807

        self.conv_head1 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.fc_head1 = nn.Linear(flattened_dim, 1)

        self.conv_head2 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.fc_head2 = nn.Linear(flattened_dim, 1)

        self.conv_head3 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.fc_head3 = nn.Linear(flattened_dim, 1)

        self.conv_head4 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.fc_head4 = nn.Linear(flattened_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        out = self.conv_b1(x)
        out = self.conv_b2(out)
        out = self.conv_b3(out)
        out = self.conv_b4(out)
        out = self.conv_b5(out)
        # print(out.shape)
        # out = self.conv6(out)
        # out = self.conv7(out)

        m1 = F.relu(self.conv_head1(out))
        m1 = self.fc_head1(m1.view((x.size()[0], -1)))
        # m1 = F.sigmoid(m1)

        m2 = F.relu(self.conv_head2(out))
        m2 = self.fc_head2(m2.view((x.size()[0], -1)))
        # m2 = F.sigmoid(m2)

        m3 = F.relu(self.conv_head3(out))
        m3 = self.fc_head3(m3.view((x.size()[0], -1)))
        # m3 = F.sigmoid(m3)

        m4 = F.relu(self.conv_head4(out))
        m4 = self.fc_head4(m4.view((x.size()[0], -1)))
        # m4 = F.sigmoid(m4)

        masked = torch.zeros_like(x[:, 0])

        # base_mask = torch.tensor(0)
        # masked += m1.unsqueeze(2) * dB_to_amplitude(x[:, 0])
        # masked += m2.unsqueeze(2) * dB_to_amplitude(x[:, 1])
        # masked += m3.unsqueeze(2) * dB_to_amplitude(x[:, 2])
        # # masked += base_mask + x[:, 3]
        # masked += m4.unsqueeze(2) * dB_to_amplitude(x[:, 3])

        masked += m1.unsqueeze(2) * x[:, 0]
        masked += m2.unsqueeze(2) * x[:, 1]
        masked += m3.unsqueeze(2) * x[:, 2]
        # masked += base_mask + x[:, 3]
        masked += m4.unsqueeze(2) * x[:, 3]

        return masked, (m1, m2, m3, m4)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    d_train = MultitrackAudioDataset(
        '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio',
        songlist=weathervane_music,
        chunk_length=1,
        seed=321,
        normalize=False
    )

    train_loader = DataLoader(d_train,
                              batch_size=8,
                              shuffle=False,
                              num_workers=6,
                              pin_memory=True,
                              drop_last=False,
                              timeout=0)

    model = MixingModelScalar2s().to(device)

    num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{} trainable parameters'.format(num_trainable_param))

    for batch in train_loader:
        model(batch['train_features'].to(device))
        break
