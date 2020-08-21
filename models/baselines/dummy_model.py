import torch
import torch.nn as nn


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


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(x[:, 0])
        # masked += x[:, 0]
        # masked += x[:, 1]
        # masked += x[:, 2]
        # masked += x[:, 3]
        masked += dB_to_amplitude(x[:, 0])
        masked += dB_to_amplitude(x[:, 1])
        masked += dB_to_amplitude(x[:, 2])
        masked += dB_to_amplitude(x[:, 3])

        return amplitude_to_dB(masked)
        # return masked
