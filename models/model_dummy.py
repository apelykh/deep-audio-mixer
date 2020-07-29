import torch
import torch.nn as nn


class ModelDummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # masked = torch.zeros_like(x[:, 0])
        # masked += 0.25 * x[:, 0]
        # masked += 0.25 * x[:, 1]
        # masked += 0.25 * x[:, 2]
        # masked += 0.25 * x[:, 3]

        masked, _ = torch.max(x, dim=1)

        return masked
